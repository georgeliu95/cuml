/*
 * Copyright (c) 2018-2022, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "multigroups/mg_accessor.cuh"
#include "adjgraph/runner.cuh"
#include "corepoints/compute.cuh"
#include "corepoints/exchange.cuh"
#include "mergelabels/runner.cuh"
#include "mergelabels/tree_reduction.cuh"
#include "vertexdeg/runner.cuh"
#include <common/nvtx.hpp>
#include <raft/core/nvtx.hpp>
#include <raft/label/classlabels.cuh>
#include <raft/sparse/csr.hpp>
#include <raft/util/cudart_utils.hpp>

#include <cuml/common/logger.hpp>

#include <raft/core/nvtx.hpp>

#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#include <thrust/iterator/counting_iterator.h>

#include <cstddef>

namespace ML {
namespace Dbscan {

static const int TPB = 256;

/**
 * Adjust labels from weak_cc primitive to match sklearn:
 * 1. Turn any labels matching MAX_LABEL into -1
 * 2. Subtract 1 from all other labels.
 */
template <typename Index_ = int>
__global__ void relabelForSkl(Index_* labels, Index_ N, Index_ MAX_LABEL)
{
  Index_ tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid < N) {
    if (labels[tid] == MAX_LABEL) {
      labels[tid] = -1;
    } else {
      --labels[tid];
    }
  }
}

/**
 * Turn the non-monotonic labels from weak_cc primitive into
 * an array of labels drawn from a monotonically increasing set.
 */
template <typename Index_ = int>
void final_relabel(Index_* db_cluster, Index_ N, cudaStream_t stream)
{
  Index_ MAX_LABEL = std::numeric_limits<Index_>::max();
  raft::label::make_monotonic(
    db_cluster, db_cluster, N, stream, [MAX_LABEL] __device__(Index_ val) {
      return val == MAX_LABEL;
    });
}

/**
 * Run the DBSCAN algorithm (common code for single-GPU and multi-GPU)
 * @tparam opg Whether we are running in a multi-node multi-GPU context
 * @param[in]  handle       raft handle
 * @param[in]  x            Input data (N*D row-major device array, or N*N for precomputed)
 * @param[in]  N            Number of points
 * @param[in]  D            Dimensionality of the points
 * @param[in]  start_row    Index of the offset for this node
 * @param[in]  n_owned_rows Number of rows (points) owned by this node
 * @param[in]  eps          Epsilon neighborhood criterion
 * @param[in]  min_pts      Core points criterion
 * @param[out] labels       Output labels (device array of length N)
 * @param[out] core_indices If not nullptr, the indices of core points are written in this array
 * @param[in]  algo_vd      Algorithm used for the vertex degrees
 * @param[in]  algo_adj     Algorithm used for the adjacency graph
 * @param[in]  algo_ccl     Algorithm used for the final relabel
 * @param[in]  workspace    Temporary global memory buffer used to store intermediate computations
 *                          If nullptr, then this function will return the workspace size needed.
 *                          It is the responsibility of the user to allocate and free this buffer!
 * @param[in]  batch_size   Batch size
 * @param[in]  stream       The CUDA stream where to launch the kernels
 * @return In case the workspace pointer is null, this returns the size needed.
 */
template <typename Type_f, typename Index_ = int, bool opg = false>
std::size_t run(const raft::handle_t& handle,
                const Type_f* x,
                Index_ N,
                Index_ D,
                Index_ start_row,
                Index_ n_owned_rows,
                Type_f eps,
                Index_ min_pts,
                Index_* labels,
                Index_* core_indices,
                int algo_vd,
                int algo_adj,
                int algo_ccl,
                void* workspace,
                std::size_t batch_size,
                cudaStream_t stream,
                raft::distance::DistanceType metric)
{
  const std::size_t align = 256;
  Index_ n_batches        = raft::ceildiv((std::size_t)n_owned_rows, batch_size);

  int my_rank;
  if (opg) {
    const auto& comm = handle.get_comms();
    my_rank          = comm.get_rank();
  } else
    my_rank = 0;

  /**
   * Note on coupling between data types:
   * - adjacency graph has a worst case size of N * batch_size elements. Thus,
   * if N is very close to being greater than the maximum 32-bit IdxType type used, a
   * 64-bit IdxType should probably be used instead.
   * - exclusive scan is the CSR row index for the adjacency graph and its values have a
   * risk of overflowing when N * batch_size becomes larger what can be stored in IdxType
   * - the vertex degree array has a worst case of each element having all other
   * elements in their neighborhood, so any IdxType can be safely used, so long as N doesn't
   * overflow.
   */
  std::size_t adj_size      = raft::alignTo<std::size_t>(sizeof(bool) * N * batch_size, align);
  std::size_t core_pts_size = raft::alignTo<std::size_t>(sizeof(bool) * N, align);
  std::size_t m_size        = raft::alignTo<std::size_t>(sizeof(bool), align);
  std::size_t vd_size       = raft::alignTo<std::size_t>(sizeof(Index_) * (batch_size + 1), align);
  std::size_t ex_scan_size  = raft::alignTo<std::size_t>(sizeof(Index_) * batch_size, align);
  std::size_t row_cnt_size  = raft::alignTo<std::size_t>(sizeof(Index_) * batch_size, align);
  std::size_t labels_size   = raft::alignTo<std::size_t>(sizeof(Index_) * N, align);

  Index_ MAX_LABEL = std::numeric_limits<Index_>::max();

  ASSERT(N * batch_size < static_cast<std::size_t>(MAX_LABEL),
         "An overflow occurred with the current choice of precision "
         "and the number of samples. (Max allowed batch size is %ld, but was %ld). "
         "Consider using double precision for the output labels.",
         (unsigned long)(MAX_LABEL / N),
         (unsigned long)batch_size);

  if (workspace == NULL) {
    auto size =
      adj_size + core_pts_size + m_size + vd_size + ex_scan_size + row_cnt_size + 2 * labels_size;
    return size;
  }

  // partition the temporary workspace needed for different stages of dbscan.

  Index_ maxadjlen  = 0;
  Index_ curradjlen = 0;
  char* temp        = (char*)workspace;
  bool* adj         = (bool*)temp;
  temp += adj_size;
  bool* core_pts = (bool*)temp;
  temp += core_pts_size;
  bool* m = (bool*)temp;
  temp += m_size;
  Index_* vd = (Index_*)temp;
  temp += vd_size;
  Index_* ex_scan = (Index_*)temp;
  temp += ex_scan_size;
  Index_* row_counters = (Index_*)temp;
  temp += row_cnt_size;
  Index_* labels_temp = (Index_*)temp;
  temp += labels_size;
  Index_* work_buffer = (Index_*)temp;
  temp += labels_size;

  // Compute the mask
  // 1. Compute the part owned by this worker (reversed order of batches to
  // keep the batch 0 in memory)
  for (int i = n_batches - 1; i >= 0; i--) {
    Index_ start_vertex_id = start_row + i * batch_size;
    Index_ n_points        = min(n_owned_rows - i * batch_size, batch_size);

    CUML_LOG_DEBUG(
      "- Batch %d / %ld (%ld samples)", i + 1, (unsigned long)n_batches, (unsigned long)n_points);

    CUML_LOG_DEBUG("--> Computing vertex degrees");
    raft::common::nvtx::push_range("Trace::Dbscan::VertexDeg");
    VertexDeg::run<Type_f, Index_>(
      handle, adj, vd, x, eps, N, D, algo_vd, start_vertex_id, n_points, stream, metric);
    raft::common::nvtx::pop_range();

    CUML_LOG_DEBUG("--> Computing core point mask");
    raft::common::nvtx::push_range("Trace::Dbscan::CorePoints");
    CorePoints::compute<Index_>(handle, vd, core_pts, min_pts, start_vertex_id, n_points, stream);
    raft::common::nvtx::pop_range();
  }
  // 2. Exchange with the other workers
  if (opg) CorePoints::exchange(handle, core_pts, N, start_row, stream);

  // Compute the labelling for the owned part of the graph
  raft::sparse::WeakCCState state(m);
  rmm::device_uvector<Index_> adj_graph(0, stream);

  for (int i = 0; i < n_batches; i++) {
    Index_ start_vertex_id = start_row + i * batch_size;
    Index_ n_points        = min(n_owned_rows - i * batch_size, batch_size);
    if (n_points <= 0) break;

    CUML_LOG_DEBUG(
      "- Batch %d / %ld (%ld samples)", i + 1, (unsigned long)n_batches, (unsigned long)n_points);

    // i==0 -> adj and vd for batch 0 already in memory
    if (i > 0) {
      CUML_LOG_DEBUG("--> Computing vertex degrees");
      raft::common::nvtx::push_range("Trace::Dbscan::VertexDeg");
      VertexDeg::run<Type_f, Index_>(
        handle, adj, vd, x, eps, N, D, algo_vd, start_vertex_id, n_points, stream, metric);
      raft::common::nvtx::pop_range();
    }
    raft::update_host(&curradjlen, vd + n_points, 1, stream);
    handle.sync_stream(stream);

    CUML_LOG_DEBUG("--> Computing adjacency graph with %ld nnz.", (unsigned long)curradjlen);
    raft::common::nvtx::push_range("Trace::Dbscan::AdjGraph");
    if (curradjlen > maxadjlen || adj_graph.data() == NULL) {
      maxadjlen = curradjlen;
      adj_graph.resize(maxadjlen, stream);
    }
    AdjGraph::run<Index_>(handle,
                          adj,
                          vd,
                          adj_graph.data(),
                          curradjlen,
                          ex_scan,
                          N,
                          algo_adj,
                          n_points,
                          row_counters,
                          stream);

    raft::common::nvtx::pop_range();

    CUML_LOG_DEBUG("--> Computing connected components");
    raft::common::nvtx::push_range("Trace::Dbscan::WeakCC");
    raft::sparse::weak_cc_batched<Index_>(
      i == 0 ? labels : labels_temp,
      ex_scan,
      adj_graph.data(),
      curradjlen,
      N,
      start_vertex_id,
      n_points,
      &state,
      stream,
      [core_pts, N] __device__(Index_ global_id) {
        return global_id < N ? __ldg((char*)core_pts + global_id) : 0;
      });
    raft::common::nvtx::pop_range();

    if (i > 0) {
      // The labels_temp array contains the labelling for the neighborhood
      // graph of the current batch. This needs to be merged with the labelling
      // created by the previous batches.
      // Using the labelling from the previous batches as initial value for
      // weak_cc_batched and skipping the merge step would lead to incorrect
      // results as described in #3094.
      CUML_LOG_DEBUG("--> Accumulating labels");
      raft::common::nvtx::push_range("Trace::Dbscan::MergeLabels");
      MergeLabels::run<Index_>(handle, labels, labels_temp, core_pts, work_buffer, m, N, stream);
      raft::common::nvtx::pop_range();
    }
  }

  // Combine the results in the multi-node multi-GPU case
  if (opg)
    MergeLabels::tree_reduction(handle, labels, labels_temp, core_pts, work_buffer, m, N, stream);

  /// TODO: optional minimalization step for border points

  // Final relabel
  if (my_rank == 0) {
    raft::common::nvtx::push_range("Trace::Dbscan::FinalRelabel");
    if (algo_ccl == 2) final_relabel(labels, N, stream);
    std::size_t nblks = raft::ceildiv<std::size_t>(N, TPB);
    relabelForSkl<Index_><<<nblks, TPB, 0, stream>>>(labels, N, MAX_LABEL);
    RAFT_CUDA_TRY(cudaPeekAtLastError());
    raft::common::nvtx::pop_range();

    // Calculate the core_indices only if an array was passed in
    if (core_indices != nullptr) {
      raft::common::nvtx::range fun_scope("Trace::Dbscan::CoreSampleIndices");

      // Create the execution policy
      auto thrust_exec_policy = handle.get_thrust_policy();

      // Get wrappers for the device ptrs
      thrust::device_ptr<bool> dev_core_pts       = thrust::device_pointer_cast(core_pts);
      thrust::device_ptr<Index_> dev_core_indices = thrust::device_pointer_cast(core_indices);

      // First fill the core_indices with -1 which will be used if core_point_count < N
      thrust::fill_n(thrust_exec_policy, dev_core_indices, N, (Index_)-1);

      auto index_iterator = thrust::counting_iterator<Index_>(0);

      // Perform stream reduction on the core points. The core_pts acts as the stencil and we use
      // thrust::counting_iterator to return the index
      thrust::copy_if(thrust_exec_policy,
                      index_iterator,
                      index_iterator + N,
                      dev_core_pts,
                      dev_core_indices,
                      [=] __device__(const bool is_core_point) { return is_core_point; });
    }
  }

  CUML_LOG_DEBUG("Done.");
  return (std::size_t)0;
}

/**
 * Run the DBSCAN algorithm (common code for multi-groups on single-GPU)
 * @tparam opg Whether we are running in a multi-node multi-GPU context (disabled for now)
 * @param[in]  handle       raft handle
 * @param[in]  x            Input data (N*D row-major device array)
 * @param[in]  N            Number of the concatenated and padding points
 * @param[in]  D            Dimensionality of the concatenated and padding points
 * @param[in]  n_groups     Number of individual groups
 * @param[in]  n_rows       Numbers of samples
 * @param[in]  n_cols       Number of features
 * @param[in]  eps          Epsilon neighborhood criterion (device array of length n_groups)
 * @param[in]  min_pts      Core points criterion (device array of length n_groups)
 * @param[out] labels       Output labels (device array of length N)
 * @param[out] core_indices If not nullptr, the indices of core points are written in this array
 * @param[in]  algo_vd      Algorithm used for the vertex degrees
 * @param[in]  algo_adj     Algorithm used for the adjacency graph
 * @param[in]  algo_ccl     Algorithm used for the final relabel
 * @param[in]  workspace    Temporary global memory buffer used to store intermediate computations
 *                          If nullptr, then this function will return the workspace size needed.
 *                          It is the responsibility of the user to allocate and free this buffer!
 * @param[in]  stream       The CUDA stream where to launch the kernels
 * @return In case the workspace pointer is null, this returns the size needed.
 */
template <typename Type_f, typename Index_ = int, bool opg = false>
std::size_t run(const raft::handle_t& handle,
                const Type_f* x,
                Index_ n_groups,
                Index_* n_rows,
                Index_ n_cols,
                Type_f* eps,
                Index_* min_pts,
                Index_* labels,
                Index_* core_indices,
                int algo_vd,
                int algo_adj,
                int algo_ccl,
                void* workspace,
                cudaStream_t stream,
                raft::distance::DistanceType metric)
{
  const std::size_t align = 256;
  const Metadata::MultiGroupMetaData<Index_> mg_metadata(n_groups, n_rows, n_cols);
  Index_ N = mg_metadata.nRowsSum;
  Index_ n_rows_max = mg_metadata.nRowsMax;
  Index_ D = n_cols;

  /**
   * Note on coupling between data types:
   * - adjacency graph has a worst case size of N * max_n_rows elements. Thus,
   * if N is very close to being greater than the maximum 32-bit IdxType type used, a
   * 64-bit IdxType should probably be used instead.
   * - exclusive scan is the CSR row index for the adjacency graph and its values have a
   * risk of overflowing when N * max_n_rows becomes larger what can be stored in IdxType
   * - the vertex degree array has a worst case of each element having all other
   * elements in their neighborhood, so any IdxType can be safely used, so long as N doesn't
   * overflow.
   */
  std::size_t metadata_size = mg_metadata.getWorkspaceSize();
  std::size_t adj_acc_size  = Metadata::AdjGraphAccessor<bool, Index_>(mg_metadata, nullptr).getWorkspaceSize();
  std::size_t eps_size      = raft::alignTo<std::size_t>(sizeof(Type_f) * n_groups, align);
  std::size_t min_pts_size  = raft::alignTo<std::size_t>(sizeof(Index_) * n_groups, align);
  // std::size_t adj_size      = raft::alignTo<std::size_t>(sizeof(bool) * N * n_rows_max, align);
  std::size_t adj_size      = Metadata::AdjGraphAccessor<bool, Index_>(mg_metadata, nullptr).getLayoutSize();
  std::size_t core_pts_size = raft::alignTo<std::size_t>(sizeof(bool) * N, align);
  std::size_t m_size        = raft::alignTo<std::size_t>(sizeof(bool), align);
  std::size_t vd_size       = raft::alignTo<std::size_t>(sizeof(Index_) * (N + n_groups + 1), align);
  std::size_t ex_scan_size  = raft::alignTo<std::size_t>(sizeof(Index_) * N, align);
  std::size_t row_cnt_size  = raft::alignTo<std::size_t>(sizeof(Index_) * N, align);
  std::size_t labels_size   = raft::alignTo<std::size_t>(sizeof(Index_) * N, align);

  Index_ MAX_LABEL = std::numeric_limits<Index_>::max();

  /// TODO: modify this part of code.
  // ASSERT(N * batch_size < static_cast<std::size_t>(MAX_LABEL),
  //        "An overflow occurred with the current choice of precision "
  //        "and the number of samples. (Max allowed batch size is %ld, but was %ld). "
  //        "Consider using double precision for the output labels.",
  //        (unsigned long)(MAX_LABEL / N),
  //        (unsigned long)batch_size);

  if (workspace == NULL) {
    auto size =
      metadata_size + adj_acc_size + eps_size + min_pts_size + adj_size + core_pts_size + m_size + vd_size + ex_scan_size + row_cnt_size + 2 * labels_size;
    return size;
  }

  // partition the temporary workspace needed for different stages of dbscan.

  Index_ maxadjlen  = 0;
  Index_ curradjlen = 0;
  char* temp        = (char*)workspace;
  Index_* metadata_buffer = (Index_*)temp;
  temp += metadata_size;
  Index_* adj_acc_buffer = (Index_*)temp;
  temp += adj_acc_size;
  Type_f* eps2 = (Type_f*)temp;
  temp += eps_size;
  Index_* min_pts2 = (Index_*)temp;
  temp += min_pts_size;
  bool* adj = (bool*)temp;
  temp += adj_size;
  bool* core_pts = (bool*)temp;
  temp += core_pts_size;
  Index_* vd = (Index_*)temp;
  temp += vd_size;
  bool* m = (bool*)temp;
  temp += m_size;
  Index_* ex_scan = (Index_*)temp;
  temp += ex_scan_size;
  Index_* row_counters = (Index_*)temp;
  temp += row_cnt_size;
  Index_* labels_temp = (Index_*)temp;
  temp += labels_size;
  Index_* work_buffer = (Index_*)temp;
  temp += labels_size;

  mg_metadata.initialize(handle, metadata_buffer, metadata_size, stream);
  Metadata::PointAccessor<Type_f, Index_> pts_accessor(mg_metadata, x);
  Metadata::VertexDegAccessor<Index_, Index_> vd_accessor(mg_metadata, vd);
  Metadata::AdjGraphAccessor<bool, Index_> adj_accessor(mg_metadata, adj);
  Metadata::CorePointAccessor<bool, Index_> corepts_accessor(mg_metadata, core_pts);
  RAFT_CUDA_TRY(cudaMemcpyAsync(eps2, eps, n_groups * sizeof(Type_f), cudaMemcpyHostToDevice, stream));
  RAFT_CUDA_TRY(cudaMemcpyAsync(min_pts2, min_pts, n_groups * sizeof(Index_), cudaMemcpyHostToDevice, stream));
  RAFT_CUDA_TRY(cudaMemsetAsync(adj, 0, adj_size + core_pts_size + vd_size, stream));
  adj_accessor.initialize(handle, adj_acc_buffer, adj_acc_size, stream);
  corepts_accessor.initialize(handle, work_buffer, labels_size, stream);

  // Compute the vertex degree
  CUML_LOG_DEBUG("--> Computing vertex degrees");
  raft::common::nvtx::push_range("Trace::Dbscan::VertexDeg");
  VertexDeg::run<Type_f, Index_>(
    handle, adj_accessor, vd_accessor, pts_accessor, eps2, algo_vd, stream, metric);
  raft::common::nvtx::pop_range();

  CUML_LOG_DEBUG("--> Computing core point mask");
  raft::common::nvtx::push_range("Trace::Dbscan::CorePoints");
  // CorePoints::compute<Index_>(handle, vd, core_pts, min_pts, start_vertex_id, n_points, stream);
  CorePoints::multi_group_compute(handle, vd_accessor, corepts_accessor, min_pts);
  raft::common::nvtx::pop_range();

  // Compute the labelling for the owned part of the graph
  raft::sparse::WeakCCState state(m);
  rmm::device_uvector<Index_> adj_graph(0, stream);

  raft::update_host(&curradjlen, vd + N, 1, stream);
  handle.sync_stream(stream);

  CUML_LOG_DEBUG("--> Computing adjacency graph with %ld nnz.", (unsigned long)curradjlen);
  raft::common::nvtx::push_range("Trace::Dbscan::AdjGraph");
  if (curradjlen > maxadjlen || adj_graph.data() == NULL) {
    maxadjlen = curradjlen;
    adj_graph.resize(maxadjlen, stream);
  }
  // AdjGraph::run<Index_>(handle,
  //                       adj,
  //                       vd,
  //                       adj_graph.data(),
  //                       curradjlen,
  //                       ex_scan,
  //                       N,
  //                       algo_adj,
  //                       n_points,
  //                       row_counters,
  //                       stream);
  raft::common::nvtx::pop_range();

  CUML_LOG_DEBUG("--> Computing connected components");
  raft::common::nvtx::push_range("Trace::Dbscan::WeakCC");
  raft::sparse::weak_cc_batched<Index_>(
    labels,
    ex_scan,
    adj_graph.data(),
    curradjlen,
    N,
    0,
    N,
    &state,
    stream,
    [core_pts, N] __device__(Index_ global_id) {
      return global_id < N ? __ldg((char*)core_pts + global_id) : 0;
    });
  raft::common::nvtx::pop_range();

  // Final relabel
  raft::common::nvtx::push_range("Trace::Dbscan::FinalRelabel");
  if (algo_ccl == 2) final_relabel(labels, N, stream);
  // std::size_t nblks = raft::ceildiv<std::size_t>(N, TPB);
  // relabelForSkl<Index_><<<nblks, TPB, 0, stream>>>(labels, N, MAX_LABEL);
  raft::common::nvtx::pop_range();

  // Calculate the core_indices only if an array was passed in
  /* if (core_indices != nullptr) {
    raft::common::nvtx::range fun_scope("Trace::Dbscan::CoreSampleIndices");

    // Create the execution policy
    auto thrust_exec_policy = handle.get_thrust_policy();

    // Get wrappers for the device ptrs
    thrust::device_ptr<bool> dev_core_pts       = thrust::device_pointer_cast(core_pts);
    thrust::device_ptr<Index_> dev_core_indices = thrust::device_pointer_cast(core_indices);

    // First fill the core_indices with -1 which will be used if core_point_count < N
    thrust::fill_n(thrust_exec_policy, dev_core_indices, N, (Index_)-1);

    auto index_iterator = thrust::counting_iterator<Index_>(0);

    // Perform stream reduction on the core points. The core_pts acts as the stencil and we use
    // thrust::counting_iterator to return the index
    thrust::copy_if(thrust_exec_policy,
                    index_iterator,
                    index_iterator + N,
                    dev_core_pts,
                    dev_core_indices,
                    [=] __device__(const bool is_core_point) { return is_core_point; });
  } */

  CUML_LOG_DEBUG("Done.");
  return (std::size_t)0;
}

}  // namespace Dbscan
}  // namespace ML
