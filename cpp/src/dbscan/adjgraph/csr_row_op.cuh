/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.
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

#include <raft/cuda_utils.cuh>
#include <raft/cudart_utils.h>
#include <raft/sparse/detail/cusparse_wrappers.h>

#include <thrust/device_ptr.h>
#include <thrust/scan.h>

#include <cuda_runtime.h>
#include <stdio.h>

#include <algorithm>
#include <iostream>

#include "pack.h"

namespace ML {
namespace Dbscan {
namespace AdjGraph {
namespace Algo {

template <typename Index_, typename BaseClass = ML::Dbscan::DataLoader<bool, Index_>>
struct CsrRowOpBathced : public BaseClass {
 public:
  __device__ CsrRowOpBathced(Index_ _n_groups,
                             Index_ _n_rows,
                             Index_ _max_rows,
                             Index_ _stride,
                             const Index_* _row_starts,
                             const Index_* _row_steps,
                             bool* _data)
    : BaseClass(_n_groups, _n_rows, _max_rows, _stride, _row_starts, _row_steps, _data)
  {
  }

};

template <typename Index_, typename Lambda = auto(Index_, Index_, Index_)->void>
static __global__ void csr_row_op_batched_kernel(Index_ n_groups,
                                                 Index_ n_rows,
                                                 Index_ max_rows,
                                                 Index_ stride,
                                                 const Index_* row_starts,
                                                 const Index_* row_steps,
                                                 bool* data,
                                                 const Index_* row_ind,
                                                 Index_* row_ind_ptr,
                                                 Index_ nnz,
                                                 Lambda op)
{
  Index_ group_id = blockIdx.y * blockDim.y + threadIdx.y;
  Index_ row      = blockIdx.x * blockDim.x + threadIdx.x;

  CsrRowOpBathced<Index_, ML::Dbscan::DataLoader<bool, Index_>> spm_ldr(
    n_groups, n_rows, max_rows, stride, row_starts, row_steps, data);

  const Index_ start_row = row_starts[group_id];
  const Index_ step      = row_steps[group_id];
  // const bool* adj        = spm_ldr.dev_ptr(group_id, start_row);
  Index_ map_row = row + start_row;

  if (row < step && map_row < n_rows && group_id < n_groups) {
    Index_ start_idx = row_ind[map_row];
    Index_ stop_idx  = map_row < n_rows - 1 ? row_ind[map_row + 1] : nnz;
    op(row, start_idx, stop_idx); /* Do nothing here */

    // printf(
    //   "gid: %ld, row: %ld, start_row: %ld, step: %ld, stride: %ld, start_idx: %ld, stop_idx:
    //   %ld\n", (long int)group_id, (long int)row, (long int)start_row, (long int)step, (long
    //   int)stride, (long int)start_idx, (long int)stop_idx);

    Index_ k = 0;
    for (Index_ i = 0; i < step; i++) {
      // if (adj[stride * i + row]) {
      if (*spm_ldr.dev_ptr(group_id, i, map_row)) {
        row_ind_ptr[start_idx + k] = i + start_row;
        k += 1;
      }
    }
  }
}

/**
 * @brief Perform a custom row operation on a CSR matrix in batches.
 * @tparam T numerical type of row_ind array
 * @tparam TPB_X number of threads per block to use for underlying kernel
 * @tparam Lambda type of custom operation function
 * @param row_ind the CSR row_ind array to perform parallel operations over
 * @param n_rows total number vertices in graph
 * @param nnz number of non-zeros
 * @param op custom row operation functor accepting the row and beginning index.
 * @param stream cuda stream to use
 */
template <typename Index_, typename Lambda = auto(Index_, Index_, Index_)->void>
static void csr_row_op_batched(Pack<Index_> data, Lambda fused_op, cudaStream_t stream)
{
  const Index_* row_ind = data.ex_scan;
  Index_ nnz            = data.adjnnz;
  Index_* row_ind_ptr   = data.adj_graph;
  auto adj_dev_ldr     = data.data_loader;

  Index_ n_groups          = adj_dev_ldr.n_groups;
  Index_ n_rows            = adj_dev_ldr.n_rows;
  const Index_* row_starts = adj_dev_ldr.row_starts;
  const Index_* row_steps  = adj_dev_ldr.row_steps;
  Index_ max_rows          = adj_dev_ldr.max_rows;
  Index_ stride            = adj_dev_ldr.stride;

  Index_ TPB_X = 32;
  if (max_rows % 256 > 127) {
    TPB_X = 256;
  } else if (max_rows % 128 > 63) {
    TPB_X = 128;
  } else if (max_rows % 64 > 31) {
    TPB_X = 64;
  }
  Index_ TPB_Y = 1;

  dim3 grid(raft::ceildiv(max_rows, Index_(TPB_X)), raft::ceildiv(n_groups, Index_(TPB_Y)), 1);
  dim3 blk(TPB_X, TPB_Y, 1);
  // csr_row_op_batched_kernel<Index_><<<grid, blk, 0, stream>>>(adj_host_ldr,
  //                                                             row_ind,
  //                                                             row_ind_ptr,
  //                                                             row_starts,
  //                                                             row_steps,
  //                                                             adj_host_ldr.stride,
  //                                                             n_rows,
  //                                                             nnz,
  //                                                             fused_op);
  csr_row_op_batched_kernel<Index_><<<grid, blk, 0, stream>>>(n_groups,
                                                              n_rows,
                                                              max_rows,
                                                              stride,
                                                              row_starts,
                                                              row_steps,
                                                              adj_dev_ldr.m_data,
                                                              row_ind,
                                                              row_ind_ptr,
                                                              nnz,
                                                              fused_op);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

template <typename Index_, int TPB_X = 32, typename Lambda = auto(Index_, Index_, Index_)->void>
void csr_adj_graph_batched(Pack<Index_> data, cudaStream_t stream)
{
  auto fused_op = ([] __device__(Index_ row, Index_ start_idx, Index_ stop_idx) {});
  csr_row_op_batched<Index_>(data, fused_op, stream);
}
}  // namespace Algo
}  // namespace AdjGraph
}  // namespace Dbscan
}  // namespace ML
