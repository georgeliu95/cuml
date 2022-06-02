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

#include "adjgraph/runner.cuh"
#include "corepoints/compute.cuh"
#include "corepoints/exchange.cuh"
#include "mergelabels/runner.cuh"
#include "mergelabels/tree_reduction.cuh"
#include "vertexdeg/runner.cuh"
#include <common/nvtx.hpp>
#include <label/classlabels.cuh>
#include <raft/common/nvtx.hpp>
#include <raft/cudart_utils.h>
#include <raft/sparse/csr.hpp>

#include <cuml/common/logger.hpp>

#include <raft/common/nvtx.hpp>

#include <label/classlabels.cuh>

#include <cstddef>

#define PRINT_TEMPVAL false

namespace {
const bool bHost_x     = false;
const bool bHost_vd    = false;
const bool bHost_cpt   = false;
const bool bHost_adj   = false;
const bool bHost_adjg  = false;
const bool bHost_adjl  = false;
const bool bHost_ex    = false;
const bool bHost_label = false;
const bool bAll        = false;
}  // namespace

namespace ML {
namespace Dbscan {

using namespace MLCommon;

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

template <typename Index_ = int>
__global__ void relabelForSklBatchedKernel(Index_* labels,
                                           const Index_* label_starts,
                                           const Index_* label_steps,
                                           const Index_* label_selects,
                                           Index_ N,
                                           Index_ n_groups,
                                           Index_ MAX_LABEL)
{
  Index_ group_id = blockIdx.y * blockDim.y + threadIdx.y;
  Index_ tid      = blockIdx.x * blockDim.x + threadIdx.x;

  Index_ start_idx = label_starts[group_id];
  Index_ step      = label_steps[group_id];
  Index_ label_val = label_selects[group_id];
  Index_ label_id  = tid + start_idx;

  if (group_id < n_groups && tid < step && label_id < N) {
    // printf(
    //   "gid: %ld, tid: %ld, start_idx: %ld, step: %ld, label_val: %ld, label_id: %ld,
    //   label:%ld\n",
    //   (long int)group_id,
    //   (long int)tid,
    //   (long int)start_idx,
    //   (long int)step,
    //   (long int)label_val,
    //   (long int)label_id,
    //   (long int)(labels[label_id]));

    if (labels[label_id] == MAX_LABEL) {
      labels[label_id] = -1;
    } else {
      labels[label_id] -= label_val;
    }
  }
}

template <typename Index_ = int>
__global__ void selectLabelStartsKernel(
  const Index_* labels, Index_* selected, const Index_* label_starts, Index_ N, Index_ MAX_LABEL)
{
  /* need coalesced access */
  Index_ tid = blockIdx.x * blockDim.x + threadIdx.x;
  while (tid < N) {
    Index_ start_idx = label_starts[tid];
    Index_ stop_idx  = label_starts[tid + 1];
    Index_ min_label = MAX_LABEL;
    for (Index_ i = 0; i < stop_idx - start_idx; ++i) {
      Index_ label_val = labels[start_idx + i];
      min_label        = (min_label < label_val) ? min_label : label_val;
    }
    selected[tid] = (min_label == MAX_LABEL) ? 0 : min_label;
    tid += gridDim.x * blockDim.x;
  }
}

template <typename Index_ = int>
__global__ void selectLabelStartsKernel(const Index_* labels,
                                        Index_* selected,
                                        const Index_* label_starts,
                                        Index_ N,
                                        Index_ n_groups,
                                        Index_ MAX_LABEL)
{
  Index_ tid     = blockIdx.x * blockDim.x + threadIdx.x;
  Index_ warp_id = tid % warpSize;
}

/**
 * Turn the non-monotonic labels from weak_cc primitive into
 * an array of labels drawn from a monotonically increasing set.
 */
template <typename Index_ = int>
void final_relabel(Index_* db_cluster, Index_ N, cudaStream_t stream)
{
  Index_ MAX_LABEL = std::numeric_limits<Index_>::max();
  MLCommon::Label::make_monotonic(
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
                cudaStream_t stream)
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
  std::size_t labels_size   = raft::alignTo<std::size_t>(sizeof(Index_) * N, align);

  Index_ MAX_LABEL = std::numeric_limits<Index_>::max();

  ASSERT(N * batch_size < static_cast<std::size_t>(MAX_LABEL),
         "An overflow occurred with the current choice of precision "
         "and the number of samples. (Max allowed batch size is %ld, but was %ld). "
         "Consider using double precision for the output labels.",
         (unsigned long)(MAX_LABEL / N),
         (unsigned long)batch_size);

  if (workspace == NULL) {
    auto size = adj_size + core_pts_size + m_size + vd_size + ex_scan_size + 2 * labels_size;
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
      handle, adj, vd, x, eps, N, D, algo_vd, start_vertex_id, n_points, stream);
    raft::common::nvtx::pop_range();

    CUML_LOG_DEBUG("--> Computing core point mask");
    raft::common::nvtx::push_range("Trace::Dbscan::CorePoints");
    CorePoints::compute<Index_>(handle, vd, core_pts, min_pts, start_vertex_id, n_points, stream);
    raft::common::nvtx::pop_range();

    if (bHost_vd && bAll) {
      std::vector<Index_> host_vd(n_points + 1, 0);
      RAFT_CUDA_TRY(cudaMemcpyAsync(
        host_vd.data(), vd, host_vd.size() * sizeof(Index_), cudaMemcpyDeviceToHost, stream));
      RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
      std::cout << "Batch " << i << " vd:" << std::endl;
      std::for_each(host_vd.begin(), host_vd.end(), [=](Index_ x) { std::cout << x << " "; });
      std::cout << std::endl;
    }
  }

  if (bHost_adj && bAll) {
    bool host_adj[N * batch_size];
    RAFT_CUDA_TRY(cudaMemcpyAsync(
      host_adj, adj, N * batch_size * sizeof(bool), cudaMemcpyDeviceToHost, stream));
    RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
    std::cout << "Adj:" << std::endl;
    for (size_t r = 0; r < batch_size; ++r) {
      for (int c = 0; c < N; ++c) {
        std::cout << host_adj[r * N + c] << " ";
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
  }

  // 2. Exchange with the other workers
  if (opg) CorePoints::exchange(handle, core_pts, N, start_row, stream);

  if (bHost_cpt && bAll) {
    bool host_corepts[N];
    RAFT_CUDA_TRY(
      cudaMemcpyAsync(host_corepts, core_pts, N * sizeof(bool), cudaMemcpyDeviceToHost, stream));
    RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
    std::cout << "core_pts:" << std::endl;
    for (auto it : host_corepts) {
      std::cout << it << " ";
    };
    std::cout << std::endl;
  }

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
        handle, adj, vd, x, eps, N, D, algo_vd, start_vertex_id, n_points, stream);
      raft::common::nvtx::pop_range();
    }
    raft::update_host(&curradjlen, vd + n_points, 1, stream);
    handle.sync_stream(stream);

    if (bHost_adjl && bAll) { std::cout << "curradjlen(" << i << ") " << curradjlen << " "; }

    CUML_LOG_DEBUG("--> Computing adjacency graph with %ld nnz.", (unsigned long)curradjlen);
    raft::common::nvtx::push_range("Trace::Dbscan::AdjGraph");
    if (curradjlen > maxadjlen || adj_graph.data() == NULL) {
      maxadjlen = curradjlen;
      adj_graph.resize(maxadjlen, stream);
    }
    AdjGraph::run<Index_>(
      handle, adj, vd, adj_graph.data(), curradjlen, ex_scan, N, algo_adj, n_points, stream);
    raft::common::nvtx::pop_range();

    if (bHost_ex && bAll) {
      std::vector<Index_> host_exs(n_points, 0);
      RAFT_CUDA_TRY(cudaMemcpyAsync(host_exs.data(),
                                    ex_scan,
                                    host_exs.size() * sizeof(Index_),
                                    cudaMemcpyDeviceToHost,
                                    stream));
      RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
      std::cout << "Batch " << i << " ex_scan:" << std::endl;
      std::for_each(host_exs.begin(), host_exs.end(), [=](Index_ x) { std::cout << x << " "; });
      std::cout << std::endl;
    }

    if (bHost_adjg && bAll) {
      std::vector<Index_> host_adjg(maxadjlen, 0);
      RAFT_CUDA_TRY(cudaMemcpyAsync(host_adjg.data(),
                                    adj_graph.data(),
                                    host_adjg.size() * sizeof(Index_),
                                    cudaMemcpyDeviceToHost,
                                    stream));
      RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
      std::cout << "Batch " << i << " adj_graph:" << std::endl;
      std::for_each(host_adjg.begin(), host_adjg.end(), [=](Index_ x) { std::cout << x << " "; });
      std::cout << std::endl;
    }

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

    if (bHost_label && bAll) {
      std::vector<Index_> host_label(N, 0);
      RAFT_CUDA_TRY(cudaMemcpyAsync(host_label.data(),
                                    (i == 0) ? labels : labels_temp,
                                    host_label.size() * sizeof(Index_),
                                    cudaMemcpyDeviceToHost,
                                    stream));
      RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
      std::cout << "Batch " << i << " labels:" << std::endl;
      std::for_each(host_label.begin(), host_label.end(), [=](Index_ x) { std::cout << x << " "; });
      std::cout << std::endl;
    }

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
    // std::cout << std::endl;
  }

  if (bHost_label && bAll) {
    std::vector<Index_> host_label(N, 0);
    RAFT_CUDA_TRY(cudaMemcpyAsync(host_label.data(),
                                  labels,
                                  host_label.size() * sizeof(Index_),
                                  cudaMemcpyDeviceToHost,
                                  stream));
    RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
    std::cout << "Labels:" << std::endl;
    std::for_each(host_label.begin(), host_label.end(), [=](Index_ x) { std::cout << x << " "; });
    std::cout << std::endl;
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

template <typename Type_f, typename Index_ = int>
std::size_t run(const raft::handle_t& handle,
                const Type_f* x,
                Index_ n_groups,
                Index_* ptr_n_rows,
                Index_ n_total_rows,
                Index_ n_cols,
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
                cudaStream_t stream)
{
  const std::size_t align = 256;
  bool divided            = batch_size < static_cast<size_t>(n_total_rows);
  // divided                 = true;
  ML::Dbscan::LookupTable<Index_> lookup_obj(n_groups, n_total_rows);

  /**
   * Use one big, disconnected graph here
   */
  std::size_t lookup_size = lookup_obj.m_size;
  std::size_t adj_size =
    divided ? raft::alignTo<std::size_t>(sizeof(bool) * n_total_rows * batch_size, align)
            : raft::alignTo<std::size_t>(sizeof(bool) * n_total_rows * n_total_rows, align);
  std::size_t core_pts_size = raft::alignTo<std::size_t>(sizeof(bool) * n_total_rows, align);
  std::size_t m_size        = raft::alignTo<std::size_t>(sizeof(bool), align);
  std::size_t vd_size =
    divided ? raft::alignTo<std::size_t>(sizeof(Index_) * (batch_size + 1), align)
            : raft::alignTo<std::size_t>(sizeof(Index_) * (n_groups + n_total_rows + 1), align);
  std::size_t ex_scan_size = divided
                               ? raft::alignTo<std::size_t>(sizeof(Index_) * batch_size, align)
                               : raft::alignTo<std::size_t>(sizeof(Index_) * n_total_rows, align);
  std::size_t labels_size  = raft::alignTo<std::size_t>(sizeof(Index_) * n_total_rows, align);

  Index_ MAX_LABEL = std::numeric_limits<Index_>::max();

  ASSERT(n_total_rows * (divided ? batch_size : n_total_rows) < static_cast<std::size_t>(MAX_LABEL),
         "An overflow occurred with the current choice of precision "
         "and the number of samples. (Max allowed batch size is %ld, but was %ld). "
         "Consider using double precision for the output labels.",
         (unsigned long)(MAX_LABEL / n_total_rows),
         (unsigned long)(divided ? batch_size : n_total_rows));

  if (workspace == NULL) {
    auto size = lookup_size + adj_size + core_pts_size + m_size + vd_size + ex_scan_size +
                ((divided) ? 3 : 1) * labels_size;
    return size;
  }

  // partition the temporary workspace needed for different stages of dbscan.

  Index_ maxadjlen         = 0;
  Index_ curradjlen        = 0;
  char* temp               = (char*)workspace;
  Index_* lookup_workspace = (Index_*)temp;
  temp += lookup_size;
  bool* adj = (bool*)temp;
  temp += adj_size;
  Index_* vd = (Index_*)temp;
  temp += vd_size;
  bool* core_pts = (bool*)temp;
  temp += core_pts_size;
  bool* m = (bool*)temp;
  temp += m_size;
  Index_* ex_scan = (Index_*)temp;
  temp += ex_scan_size;

  Index_* labels_temp = (divided) ? (Index_*)temp : nullptr;
  temp += (divided) ? labels_size : 0;
  Index_* labels_group = (divided) ? (Index_*)temp : nullptr;
  temp += (divided) ? labels_size : 0;

  Index_* work_buffer = (Index_*)temp;
  temp += labels_size;

  Index_ N = n_total_rows;
  Index_ D = n_cols;

  lookup_obj.init(handle, ptr_n_rows, lookup_workspace, stream);

#if PRINT_TEMPVAL
  if (bHost_x) {
    std::cout << "n_rows = ";
    for (int b = 0; b < n_groups; ++b) {
      std::cout << ptr_n_rows[b] << " ";
    }
    std::cout << std::endl;
    std::vector<Type_f> host_x(n_total_rows * n_cols, 0);
    RAFT_CUDA_TRY(cudaMemcpyAsync(
      host_x.data(), x, host_x.size() * sizeof(Type_f), cudaMemcpyDeviceToHost, stream));
    RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
    std::cout << "Input:" << std::endl;
    std::for_each(host_x.begin(), host_x.end(), [=](double x) { std::cout << x << " "; });
    std::cout << std::endl;
  }
#endif

  CUML_LOG_INFO("Apply divided version? %d", (int)divided);
  if (divided) {
    Index_ start_row = 0;
    for (int b = 0; b < n_groups; ++b) {
      Index_ n_rows_per_group = ptr_n_rows[b];
      Index_ n_batches        = raft::ceildiv((std::size_t)n_rows_per_group, batch_size);
      Index_ low_bound        = start_row;
      Index_ high_bound       = start_row + n_rows_per_group;

      for (int i = n_batches - 1; i >= 0; i--) {
        Index_ start_vertex_id = start_row + i * batch_size;
        Index_ n_points        = min(n_rows_per_group - i * batch_size, batch_size);

        CUML_LOG_DEBUG("- Group %d batch %d / %ld (%ld samples)",
                       b + 1,
                       i + 1,
                       (unsigned long)n_batches,
                       (unsigned long)n_points);

        CUML_LOG_DEBUG("--> Computing vertex degrees");
        raft::common::nvtx::push_range("Trace::Dbscan::VertexDeg");
        VertexDeg::run_batched<Type_f, Index_>(handle,
                                               adj,
                                               vd,
                                               nullptr,
                                               nullptr,
                                               x,
                                               eps,
                                               N,
                                               D,
                                               algo_vd,
                                               start_vertex_id,
                                               n_points,
                                               stream);
        // VertexDeg::run<Type_f, Index_>(
        //   handle, adj, vd, x, eps, N, D, algo_vd, start_vertex_id, n_points, stream);
        raft::common::nvtx::pop_range();

        if (bHost_vd && bAll) {
          std::vector<Index_> host_vd(n_points + 1, 0);
          RAFT_CUDA_TRY(cudaMemcpyAsync(
            host_vd.data(), vd, host_vd.size() * sizeof(Index_), cudaMemcpyDeviceToHost, stream));
          RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
          std::cout << "Group " << b << " batch " << i << " vd:" << std::endl;
          std::for_each(host_vd.begin(), host_vd.end(), [=](Index_ x) { std::cout << x << " "; });
          std::cout << std::endl;
        }

        CUML_LOG_DEBUG("--> Computing core point mask");
        raft::common::nvtx::push_range("Trace::Dbscan::CorePoints");
        CorePoints::compute<Index_>(
          handle, vd, core_pts, min_pts, start_vertex_id, n_points, stream);
        raft::common::nvtx::pop_range();
      }

      if (bHost_cpt && bAll) {
        int size = n_total_rows;
        bool host_corepts[size];
        RAFT_CUDA_TRY(cudaMemcpyAsync(
          host_corepts, core_pts, size * sizeof(bool), cudaMemcpyDeviceToHost, stream));
        RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
        std::cout << "core_pts:" << std::endl;
        for (int i = 0; i < size; ++i) {
          std::cout << host_corepts[i] << " ";
        };
        std::cout << std::endl;
      }
      start_row += n_rows_per_group;
    }

    start_row = 0;
    for (int b = 0; b < n_groups; ++b) {
      Index_ n_rows_per_group = ptr_n_rows[b];
      Index_ n_batches        = raft::ceildiv((std::size_t)n_rows_per_group, batch_size);
      Index_ low_bound        = start_row;
      Index_ high_bound       = start_row + n_rows_per_group;
      RAFT_CUDA_TRY(cudaMemsetAsync(adj, 0, adj_size, stream));

      // Compute the labelling for the owned part of the graph
      raft::sparse::WeakCCState state(m);
      rmm::device_uvector<Index_> adj_graph(0, stream);

      for (int i = 0; i < n_batches; i++) {
        Index_ start_vertex_id = start_row + i * batch_size;
        Index_ n_points        = min(n_rows_per_group - i * batch_size, batch_size);
        if (n_points <= 0) break;

        CUML_LOG_DEBUG("- Group %d batch %d / %ld (%ld samples)",
                       b + 1,
                       i + 1,
                       (unsigned long)n_batches,
                       (unsigned long)n_points);

        // i==0 -> adj and vd for batch 0 already in memory
        if (i >= 0) {
          CUML_LOG_DEBUG("--> Computing vertex degrees");
          raft::common::nvtx::push_range("Trace::Dbscan::VertexDeg");
          VertexDeg::run_batched<Type_f, Index_>(handle,
                                                 adj,
                                                 vd,
                                                 nullptr,
                                                 nullptr,
                                                 x,
                                                 eps,
                                                 N,
                                                 D,
                                                 algo_vd,
                                                 start_vertex_id,
                                                 n_points,
                                                 stream);
          // VertexDeg::run<Type_f, Index_>(
          //   handle, adj, vd, x, eps, N, D, algo_vd, start_vertex_id, n_points, stream);
          raft::common::nvtx::pop_range();
        }
        raft::update_host(&curradjlen, vd + n_points, 1, stream);
        handle.sync_stream(stream);

        if (bHost_adj && bAll) {
          bool host_adj[N * batch_size];
          RAFT_CUDA_TRY(cudaMemcpyAsync(
            host_adj, adj, N * batch_size * sizeof(bool), cudaMemcpyDeviceToHost, stream));
          RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
          std::cout << "Adj:" << std::endl;
          for (size_t r = 0; r < batch_size; ++r) {
            for (int c = 0; c < N; ++c) {
              std::cout << host_adj[r * N + c] << " ";
            }
            std::cout << std::endl;
          }
          std::cout << std::endl;
        }

        if (bHost_adjl && bAll) { std::cout << "curradjlen(" << i << ") " << curradjlen << " "; }

        CUML_LOG_DEBUG("--> Computing adjacency graph with %ld nnz.", (unsigned long)curradjlen);
        raft::common::nvtx::push_range("Trace::Dbscan::AdjGraph");
        if (curradjlen > maxadjlen || adj_graph.data() == NULL) {
          maxadjlen = curradjlen;
          adj_graph.resize(maxadjlen, stream);
        }
        AdjGraph::run<Index_>(
          handle, adj, vd, adj_graph.data(), curradjlen, ex_scan, N, algo_adj, n_points, stream);
        raft::common::nvtx::pop_range();

        if (bHost_ex && bAll) {
          std::vector<Index_> host_exs(n_points, 0);
          RAFT_CUDA_TRY(cudaMemcpyAsync(host_exs.data(),
                                        ex_scan,
                                        host_exs.size() * sizeof(Index_),
                                        cudaMemcpyDeviceToHost,
                                        stream));
          RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
          std::cout << "Group " << b << " batch " << i << " ex_scan:" << std::endl;
          std::for_each(host_exs.begin(), host_exs.end(), [=](Index_ x) { std::cout << x << " "; });
          std::cout << std::endl;
        }

        if (bHost_adjg && bAll) {
          std::vector<Index_> host_adjg(maxadjlen, 0);
          RAFT_CUDA_TRY(cudaMemcpyAsync(host_adjg.data(),
                                        adj_graph.data(),
                                        host_adjg.size() * sizeof(Index_),
                                        cudaMemcpyDeviceToHost,
                                        stream));
          RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
          std::cout << "Group " << b << " batch " << i << " adj_graph:" << std::endl;
          std::for_each(
            host_adjg.begin(), host_adjg.end(), [=](Index_ x) { std::cout << x << " "; });
          std::cout << std::endl;
        }

        if (bHost_cpt && bAll) {
          bool host_corepts[n_total_rows];
          RAFT_CUDA_TRY(cudaMemcpyAsync(
            host_corepts, core_pts, n_total_rows * sizeof(bool), cudaMemcpyDeviceToHost, stream));
          RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
          std::cout << "core_pts before labelling:" << std::endl;
          for (auto it : host_corepts) {
            std::cout << it << " ";
          };
          std::cout << std::endl;
        }

        CUML_LOG_DEBUG("--> Computing connected components");
        raft::common::nvtx::push_range("Trace::Dbscan::WeakCC");
        raft::sparse::weak_cc_batched<Index_>(
          i == 0 ? (b == 0 ? labels : labels_group) : labels_temp,
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

        if (bHost_label && bAll) {
          Index_* d_label = (i == 0) ? (b == 0 ? labels : labels_group) : labels_temp;
          std::vector<Index_> host_label(N, 0);
          RAFT_CUDA_TRY(cudaMemcpyAsync(host_label.data(),
                                        d_label,
                                        host_label.size() * sizeof(Index_),
                                        cudaMemcpyDeviceToHost,
                                        stream));
          RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
          std::cout << "Group " << b << " batch " << i << " ptr = " << d_label
                    << " label:" << std::endl;
          std::for_each(
            host_label.begin(), host_label.end(), [=](Index_ x) { std::cout << x << " "; });
          std::cout << std::endl;
        }

        if (i > 0) {
          // The labels_temp array contains the labelling for the neighborhood
          // graph of the current batch. This needs to be merged with the labelling
          // created by the previous batches.
          // Using the labelling from the previous batches as initial value for
          // weak_cc_batched and skipping the merge step would lead to incorrect
          // results as described in #3094.
          CUML_LOG_DEBUG("--> Accumulating labels");
          raft::common::nvtx::push_range("Trace::Dbscan::MergeLabels");
          MergeLabels::run<Index_>(handle,
                                   (b == 0 ? labels : labels_group),
                                   labels_temp,
                                   core_pts,
                                   work_buffer,
                                   m,
                                   N,
                                   stream);
          raft::common::nvtx::pop_range();
        }

        if (bHost_label && bAll) {
          Index_* d_label = b == 0 ? labels : labels_group;
          std::vector<Index_> host_label(N, 0);
          RAFT_CUDA_TRY(cudaMemcpyAsync(host_label.data(),
                                        d_label,
                                        host_label.size() * sizeof(Index_),
                                        cudaMemcpyDeviceToHost,
                                        stream));
          RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
          std::cout << "Group " << b << " batch " << i << " ptr = " << d_label
                    << " label:" << std::endl;
          std::for_each(
            host_label.begin(), host_label.end(), [=](Index_ x) { std::cout << x << " "; });
          std::cout << std::endl;
        }
      }

      if (b > 0) {
        raft::sparse::WeakCCState state(m);
        raft::common::nvtx::push_range("Trace::Dbscan::MergeLabels");
        MergeLabels::run<Index_>(handle, labels, labels_group, core_pts, work_buffer, m, N, stream);
        raft::common::nvtx::pop_range();
      }

      start_row += n_rows_per_group;
    }
  } else {
    Index_ adj_stride      = N;
    RAFT_CUDA_TRY(cudaMemsetAsync(adj, 0, adj_size + vd_size, stream));
    for (int i = 0; i < n_groups; ++i) {
      Index_ start_vertex_id = 0;
      Index_ start_row = lookup_obj.host_row_starts[i];
      Index_ n_points  = lookup_obj.host_row_steps[i];
      // Index_ n_points   = ptr_n_rows[i];

      CUML_LOG_DEBUG(
        "- Batch %d / %ld (%ld samples)", i + 1, (unsigned long)n_groups, (unsigned long)n_points);

      CUML_LOG_DEBUG("--> Computing vertex degrees");
      Index_* vd_base      = vd + start_row;
      Index_* vd_all       = vd + n_total_rows;
      Index_* vd_batch     = vd + n_total_rows + 1 + i;
      bool* adj_base       = adj + start_row * adj_stride;
      const Type_f* x_base = x + start_row * D;
      raft::common::nvtx::push_range("Trace::Dbscan::VertexDeg");
      VertexDeg::run_batched<Type_f, Index_>(handle,
                                             adj_base,
                                             vd_base,
                                             vd_batch,
                                             vd_all,
                                             x_base,
                                             eps,
                                             N,
                                             D,
                                             algo_vd,
                                             start_vertex_id,
                                             n_points,
                                             stream,
                                             i,
                                             adj_stride,
                                             &lookup_obj);
      raft::common::nvtx::pop_range();

#if PRINT_TEMPVAL
      if (bHost_vd) {
        std::vector<Index_> host_vd(n_points, 0);
        RAFT_CUDA_TRY(cudaMemcpyAsync(host_vd.data(),
                                      vd_base,
                                      host_vd.size() * sizeof(Index_),
                                      cudaMemcpyDeviceToHost,
                                      stream));
        RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
        std::cout << "Batch " << i << " vd:" << std::endl;
        std::for_each(host_vd.begin(), host_vd.end(), [=](Index_ x) { std::cout << x << " "; });
        std::cout << std::endl;
      }
#endif

      start_vertex_id += n_points;
    }

#if PRINT_TEMPVAL
    if (bHost_vd) {
      std::vector<Index_> host_vd(n_groups + n_total_rows + 1, 0);
      RAFT_CUDA_TRY(cudaMemcpyAsync(
        host_vd.data(), vd, host_vd.size() * sizeof(Index_), cudaMemcpyDeviceToHost, stream));
      RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
      std::cout << "All vd:" << std::endl;
      std::for_each(host_vd.begin(), host_vd.end(), [=](Index_ x) { std::cout << x << " "; });
      std::cout << std::endl;
    }
#endif

    CUML_LOG_DEBUG("--> Computing core point mask");
    raft::common::nvtx::push_range("Trace::Dbscan::CorePoints");
    CorePoints::compute<Index_>(handle, vd, core_pts, min_pts, 0, n_total_rows, stream);
    raft::common::nvtx::pop_range();

#if PRINT_TEMPVAL
    if (bHost_cpt) {
      int size = n_total_rows;
      bool host_corepts[size];
      RAFT_CUDA_TRY(cudaMemcpyAsync(
        host_corepts, core_pts, size * sizeof(bool), cudaMemcpyDeviceToHost, stream));
      RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
      std::cout << "core_pts:" << std::endl;
      for (int i = 0; i < size; ++i) {
        std::cout << host_corepts[i] << " ";
      };
      std::cout << std::endl;
    }
#endif

#if PRINT_TEMPVAL
    if (bHost_adj) {
      bool host_adj[N * n_total_rows];
      RAFT_CUDA_TRY(cudaMemcpyAsync(
        host_adj, adj, N * n_total_rows * sizeof(bool), cudaMemcpyDeviceToHost, stream));
      RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
      std::cout << "Adj:" << std::endl;
      for (Index_ r = 0; r < n_total_rows; ++r) {
        for (Index_ c = 0; c < N; ++c) {
          std::cout << host_adj[r * N + c] << " ";
        }
        std::cout << std::endl;
      }
      std::cout << std::endl;
    }
#endif

    // Compute the labelling for the owned part of the graph
    raft::sparse::WeakCCState state(m);
    rmm::device_uvector<Index_> adj_graph(0, stream);

    {
      Index_ n_points = n_total_rows;
      raft::update_host(&curradjlen, (vd + n_total_rows), 1, stream);
      handle.sync_stream(stream);

      if (bHost_adjl && bAll) { std::cout << "curradjlen " << curradjlen << " " << std::endl; }

      CUML_LOG_DEBUG("--> Computing adjacency graph with %ld nnz.", (unsigned long)curradjlen);
      raft::common::nvtx::push_range("Trace::Dbscan::AdjGraph");
      if (curradjlen > maxadjlen || adj_graph.data() == NULL) {
        maxadjlen = curradjlen;
        adj_graph.resize(maxadjlen, stream);
      }
      AdjGraph::run_batched<Index_>(handle,
                                    adj,
                                    vd,
                                    adj_graph.data(),
                                    curradjlen,
                                    ex_scan,
                                    N,
                                    algo_adj,
                                    n_points,
                                    stream,
                                    N,
                                    &lookup_obj);
      raft::common::nvtx::pop_range();
      CUML_LOG_DEBUG("AdjGraph done");

#if PRINT_TEMPVAL
      if (bHost_ex) {
        std::vector<Index_> host_exs(n_points, 0);
        RAFT_CUDA_TRY(cudaMemcpyAsync(host_exs.data(),
                                      ex_scan,
                                      host_exs.size() * sizeof(Index_),
                                      cudaMemcpyDeviceToHost,
                                      stream));
        RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
        std::cout << "Ex_scan:" << std::endl;
        std::for_each(host_exs.begin(), host_exs.end(), [=](Index_ x) { std::cout << x << " "; });
        std::cout << std::endl;
      }
#endif

#if PRINT_TEMPVAL
      if (bHost_adjg) {
        std::vector<Index_> host_adjg(maxadjlen, 0);
        RAFT_CUDA_TRY(cudaMemcpyAsync(host_adjg.data(),
                                      adj_graph.data(),
                                      host_adjg.size() * sizeof(Index_),
                                      cudaMemcpyDeviceToHost,
                                      stream));
        RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
        std::cout << "Adj_graph:" << std::endl;
        std::for_each(host_adjg.begin(), host_adjg.end(), [=](Index_ x) { std::cout << x << " "; });
        std::cout << std::endl;
      }
#endif

      Index_ start_vertex_id = 0;
      CUML_LOG_DEBUG("--> Computing connected components");
      raft::common::nvtx::push_range("Trace::Dbscan::WeakCC");
      raft::sparse::weak_cc_batched<Index_>(
        labels,
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
    }

#if PRINT_TEMPVAL
    if (bHost_label) {
      std::vector<Index_> host_label(n_total_rows, 0);
      RAFT_CUDA_TRY(cudaMemcpyAsync(host_label.data(),
                                    labels,
                                    host_label.size() * sizeof(Index_),
                                    cudaMemcpyDeviceToHost,
                                    stream));
      RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
      std::cout << "Labels:" << std::endl;
      std::for_each(host_label.begin(), host_label.end(), [=](Index_ x) { std::cout << x << " "; });
      std::cout << std::endl;
    }
#endif
  }

#if PRINT_TEMPVAL
  if (bHost_label) {
    std::vector<Index_> host_label(n_total_rows, 0);
    RAFT_CUDA_TRY(cudaMemcpyAsync(host_label.data(),
                                  labels,
                                  host_label.size() * sizeof(Index_),
                                  cudaMemcpyDeviceToHost,
                                  stream));
    RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
    std::cout << "Labels raw:" << std::endl;
    std::for_each(host_label.begin(), host_label.end(), [=](Index_ x) { std::cout << x << " "; });
    std::cout << std::endl;
  }
#endif

  // Final relabel
  {
    raft::common::nvtx::push_range("Trace::Dbscan::FinalRelabel");
    if (algo_ccl == 2) {
      // Index_ start_vertex_id = 0;
      // for (int i = 0; i < n_groups; ++i) {
      //   Index_ n_points = ptr_n_rows[i];
      //   final_relabel(labels + start_vertex_id, n_points, stream);
      //   start_vertex_id += n_points;
      // }
      final_relabel(labels, n_total_rows, stream);
    }

#if PRINT_TEMPVAL
    // if (bHost_label) {
    //   std::vector<Index_> host_label(n_total_rows, 0);
    //   RAFT_CUDA_TRY(cudaMemcpyAsync(host_label.data(),
    //                                 labels,
    //                                 host_label.size() * sizeof(Index_),
    //                                 cudaMemcpyDeviceToHost,
    //                                 stream));
    //   RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
    //   std::cout << "Labels after final_relabel:" << std::endl;
    //   std::for_each(host_label.begin(), host_label.end(), [=](Index_ x) { std::cout << x << " ";
    //   }); std::cout << std::endl;
    // }
#endif

    size_t TPB_X               = 32;
    size_t TPB_Y               = 4;
    const Index_* label_starts = lookup_obj.dev_row_starts;
    const Index_* label_steps  = lookup_obj.dev_row_steps;
    const Index_ max_rows      = lookup_obj.max_rows;
    Index_* label_selects      = work_buffer;

    selectLabelStartsKernel<Index_>
      <<<1, TPB_X, 0, stream>>>(labels, label_selects, label_starts, n_groups, MAX_LABEL);

    dim3 grid(raft::ceildiv(max_rows, Index_(TPB_X)), raft::ceildiv(n_groups, Index_(TPB_Y)), 1);
    dim3 blk(TPB_X, TPB_Y, 1);
    relabelForSklBatchedKernel<Index_><<<grid, blk, 0, stream>>>(
      labels, label_starts, label_steps, label_selects, N, n_groups, MAX_LABEL);

    //   std::size_t nblks = raft::ceildiv<std::size_t>(N, TPB);
    // relabelForSkl<Index_><<<nblks, TPB, 0, stream>>>(labels, N, MAX_LABEL);
    RAFT_CUDA_TRY(cudaPeekAtLastError());
    raft::common::nvtx::pop_range();

#if PRINT_TEMPVAL
    if (bHost_label) {
      std::vector<Index_> host_label(n_total_rows, 0);
      RAFT_CUDA_TRY(cudaMemcpyAsync(host_label.data(),
                                    labels,
                                    host_label.size() * sizeof(Index_),
                                    cudaMemcpyDeviceToHost,
                                    stream));
      RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
      std::cout << "Labels after relabelForSkl_relabel:" << std::endl;
      std::for_each(host_label.begin(), host_label.end(), [=](Index_ x) { std::cout << x << " "; });
      std::cout << std::endl;
    }
#endif

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

}  // namespace Dbscan
}  // namespace ML
