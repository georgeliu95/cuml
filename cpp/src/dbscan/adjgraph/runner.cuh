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

#include "algo.cuh"
#include "naive.cuh"
#include "pack.h"

namespace ML {
namespace Dbscan {
namespace AdjGraph {

template <typename Index_ = int>
void run(const raft::handle_t& handle,
         bool* adj,
         Index_* vd,
         Index_* adj_graph,
         Index_ adjnnz,
         Index_* ex_scan,
         Index_ N,
         int algo,
         Index_ batch_size,
         cudaStream_t stream)
{
  Pack<Index_> data = {vd, adj, adj_graph, adjnnz, ex_scan, N};
  switch (algo) {
    case 0: Naive::launcher<Index_>(handle, data, batch_size, stream); break;
    case 1: Algo::launcher<Index_>(handle, data, batch_size, stream); break;
    default: ASSERT(false, "Incorrect algo passed! '%d'", algo);
  }
}

template <typename Index_ = int, typename LookupType = ML::Dbscan::LookupTable<Index_>>
void run_batched(const raft::handle_t& handle,
                 bool* adj,
                 Index_* vd,
                 Index_* adj_graph,
                 Index_ adj_graph_nnz,
                 Index_* ex_scan,
                 Index_ N,
                 int algo,
                 Index_ batch_size,
                 cudaStream_t stream,
                 Index_ adj_nnz            = 0,
                 const LookupType* const lookup = nullptr)
{
  Index_ n_groups    = lookup->n_groups;
  Index_ n_rows      = lookup->n_rows;
  Index_ max_rows    = lookup->max_rows;
  Index_ stride      = adj_nnz;
  Index_* row_starts = (algo == 0) ? lookup->host_row_starts : lookup->dev_row_starts;
  Index_* row_steps  = (algo == 0) ? lookup->host_row_steps : lookup->dev_row_steps;
  ML::Dbscan::DataLoader<bool, Index_> adj_data_loader(
    n_groups, n_rows, max_rows, stride, row_starts, row_steps, adj);
  Pack<Index_> data = {vd, adj, adj_graph, adj_graph_nnz, ex_scan, N, adj_data_loader};
  switch (algo) {
    case 0: Naive::launcher_batched<Index_>(handle, data, batch_size, stream); break;
    case 1: Algo::launcher_batched<Index_>(handle, data, batch_size, stream); break;
    default: ASSERT(false, "Incorrect algo passed! '%d'", algo);
  }
}

}  // namespace AdjGraph
}  // namespace Dbscan
}  // namespace ML
