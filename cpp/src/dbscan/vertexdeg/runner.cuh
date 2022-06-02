/*
 * Copyright (c) 2018-2021, NVIDIA CORPORATION.
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

#include "algo.cuh"
#include "naive.cuh"
#include "pack.h"
#include "precomputed.cuh"

namespace ML {
namespace Dbscan {
namespace VertexDeg {

template <typename Type_f, typename Index_ = int>
void run(const raft::handle_t& handle,
         bool* adj,
         Index_* vd,
         const Type_f* x,
         Type_f eps,
         Index_ N,
         Index_ D,
         int algo,
         Index_ start_vertex_id,
         Index_ batch_size,
         cudaStream_t stream)
{
  Pack<Type_f, Index_> data = {vd, adj, x, eps, N, D};
  switch (algo) {
    case 0: Naive::launcher<Type_f, Index_>(data, start_vertex_id, batch_size, stream); break;
    case 1:
      Algo::launcher<Type_f, Index_>(handle, data, start_vertex_id, batch_size, stream);
      break;
    case 2:
      Precomputed::launcher<Type_f, Index_>(handle, data, start_vertex_id, batch_size, stream);
      break;
    default: ASSERT(false, "Incorrect algo passed! '%d'", algo);
  }
}

template <typename Type_f, typename Index_ = int, typename LookupType = ML::Dbscan::LookupTable<Index_>>
void run_batched(const raft::handle_t& handle,
                 bool* adj,
                 Index_* vd,
                 Index_* vd_batch,
                 Index_* vd_all,
                 const Type_f* x,
                 Type_f eps,
                 Index_ N,
                 Index_ D,
                 int algo,
                 Index_ start_vertex_id,
                 Index_ batch_size,
                 cudaStream_t stream,
                 Index_ group_id           = 0,
                 Index_ adj_nnz            = 0,
                 const LookupType* const lookup = nullptr)
{
  Index_ n_groups    = lookup->n_groups;
  Index_ n_rows      = lookup->n_rows;
  Index_ max_rows    = lookup->max_rows;
  Index_ adj_stride  = adj_nnz;
  Index_* row_starts = lookup->host_row_starts;
  Index_* row_steps  = lookup->host_row_steps;
  ML::Dbscan::DataLoader<bool, Index_> data_loader(
    n_groups, n_rows, max_rows, adj_stride, row_starts, row_steps, adj);
  BatchedPack<Type_f, Index_> data = {
    vd, vd_batch, vd_all, adj, x, eps, N, D, group_id, data_loader};
  switch (algo) {
    case 0:
      Naive::launcher_batched_new<Type_f, Index_>(data, start_vertex_id, batch_size, stream);
      break;
    case 1:
      Algo::launcher_batched_new<Type_f, Index_>(handle, data, start_vertex_id, batch_size, stream);
      break;
      // case 2:
      //     Precomputed::launcher<Type_f, Index_>(handle, data, start_vertex_id, batch_size,
      //     stream); break;
      break;
    default: ASSERT(false, "Incorrect algo passed! '%d'", algo);
  }
}

}  // namespace VertexDeg
}  // namespace Dbscan
}  // namespace ML
