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

#include <thrust/device_ptr.h>
#include <thrust/scan.h>

#include <cuml/common/logger.hpp>
#include <cuml/common/pinned_host_vector.hpp>

#include <raft/common/nvtx.hpp>
#include <raft/cuda_utils.cuh>
#include <raft/cudart_utils.h>
#include <raft/handle.hpp>

#pragma once

namespace ML {
namespace Dbscan {

template <typename T, typename Index_>
struct DataLoader {
  DataLoader() = default;
  __host__ __device__ __forceinline__ DataLoader(Index_ n_groups,
                                                 Index_ n_rows,
                                                 Index_ max_rows,
                                                 Index_ stride,
                                                 const Index_* row_starts,
                                                 const Index_* row_steps,
                                                 T* data)
    : n_groups(n_groups),
      n_rows(n_rows),
      max_rows(max_rows),
      stride(stride),
      row_starts(row_starts),
      row_steps(row_steps),
      m_data(data)
  {
  }

  __device__ __forceinline__ T* dev_ptr(Index_ group_id) const
  {
    return (group_id < n_groups)? m_data + row_starts[group_id] * stride : nullptr;
  }

  __device__ __forceinline__ T* dev_ptr(Index_ group_id, Index_ group_rowid) const
  {
    return (group_id < n_groups)? m_data + (row_starts[group_id] + group_rowid) * stride : nullptr;
  }

  __device__ __forceinline__ T* dev_ptr(Index_ group_id,
                                        Index_ group_rowid,
                                        Index_ group_colid) const
  {
    return (group_id < n_groups)? m_data + (row_starts[group_id] + group_rowid) * stride + group_colid : nullptr;
  }

  __host__ __forceinline__ T* host_ptr(Index_ group_id) const
  {
    return (group_id < n_groups)? m_data + row_starts[group_id] * stride : nullptr;
  }

  __host__ __forceinline__ T* host_ptr(Index_ group_id, Index_ group_rowid) const
  {
    if (group_id >= n_groups) { return nullptr; }
    Index_ n_rows_group = row_steps[group_id];
    if (group_rowid >= n_rows_group) { return nullptr; }
    return m_data + (row_starts[group_id] + group_rowid) * stride;
  }

  __host__ __forceinline__ T* host_ptr(Index_ group_id,
                                       Index_ group_rowid,
                                       Index_ group_colid) const
  {
    if (group_id >= n_groups || group_colid >= stride) { return nullptr; }
    Index_ n_rows_group = row_steps[group_id];
    if (group_rowid >= n_rows_group) { return nullptr; }
    return m_data + (row_starts[group_id] + group_rowid) * stride + group_colid;
  }

  Index_ tmp = 0;
  Index_ n_groups;
  Index_ n_rows;
  Index_ max_rows;
  Index_ stride;
  // Index_* dev_row_starts  = nullptr;
  // Index_* dev_row_steps   = nullptr;
  // Index_* host_row_starts = nullptr;
  // Index_* host_row_steps  = nullptr;
  const Index_* row_starts = nullptr;
  const Index_* row_steps  = nullptr;
  T* m_data;
};

template <typename Index_ = int>
struct LookupTable {
  LookupTable() = delete;
  LookupTable(Index_ _n_groups, Index_ _n_rows) : n_groups(_n_groups), n_rows(_n_rows)
  {
    this->size();
  }

  void init(const raft::handle_t& handle, Index_* _row_ind, Index_* workspace, cudaStream_t stream)
  {
    this->dev_row_steps = workspace;
    this->dev_row_starts =
      reinterpret_cast<Index_*>(reinterpret_cast<char*>(workspace) + m_size / 2);
    this->resetArray(stream);

    vec_row_steps.resize(n_groups + 1);
    for (int i = 0; i < n_groups; ++i) {
      vec_row_steps[i] = _row_ind[i];
      max_rows         = (max_rows < _row_ind[i]) ? _row_ind[i] : max_rows;
    }
    raft::update_device(this->dev_row_steps, vec_row_steps.data(), n_groups, stream);
    handle.sync_stream(stream);

    using namespace thrust;
    device_ptr<Index_> d_row_ind = device_pointer_cast(this->dev_row_steps);
    device_ptr<Index_> d_ex_scan = device_pointer_cast(this->dev_row_starts);
    exclusive_scan(handle.get_thrust_policy(), d_row_ind, d_row_ind + (n_groups + 1), d_ex_scan);
    vec_row_starts.resize(n_groups + 1);
    raft::update_host(vec_row_starts.data(), dev_row_starts, n_groups + 1, stream);
    handle.sync_stream(stream);

    // std::cout << "row_ind " << dev_row_steps << std::endl;
    // for (auto it : vec_row_steps) {
    //   std::cout << it << " ";
    // }
    // std::cout << std::endl;
    // std::cout << "ex_scan " << dev_row_starts << std::endl;
    // for (auto it : vec_row_starts) {
    //   std::cout << it << " ";
    // }
    // std::cout << std::endl;
    host_row_steps  = vec_row_steps.data();
    host_row_starts = vec_row_starts.data();
    return;
  }

  void resetArray(cudaStream_t stream)
  {
    RAFT_CUDA_TRY(cudaMemsetAsync(this->dev_row_steps, 0, m_size, stream));
  }

  void size()
  {
    const std::size_t align = 256;
    m_size = 2 * raft::alignTo<std::size_t>(sizeof(Index_) * (n_groups + 1), align);
  }

  Index_ n_groups    = 0;
  Index_ n_rows      = 0;
  Index_ max_rows    = 0;
  std::size_t m_size = 0;
  ML::pinned_host_vector<Index_> vec_row_steps{ML::pinned_host_vector<Index_>()};
  ML::pinned_host_vector<Index_> vec_row_starts{ML::pinned_host_vector<Index_>()};
  Index_* dev_row_steps   = nullptr;
  Index_* dev_row_starts  = nullptr;
  Index_* host_row_steps  = nullptr;
  Index_* host_row_starts = nullptr;
};

// template <typename T, typename Index_>
// struct DataLoader {
//   DataLoader() = default;
//   // __host__ __device__ DataLoader(const ML::Dbscan::LookupTable<Index_>* lookup,
//   //                                int _stride,
//   //                                T* data)
//   //   : n_groups(lookup->n_groups),
//   //     n_rows(lookup->n_rows),
//   //     max_rows(lookup->max_rows),
//   //     stride(_stride),
//   //     dev_row_starts(lookup->dev_row_starts),
//   //     dev_row_steps(lookup->dev_row_steps),
//   //     host_row_starts(lookup->host_row_starts),
//   //     host_row_steps(lookup->host_row_steps),
//   //     m_data(data)
//   // {
//   // }
//   __host__ __device__ DataLoader(Index_ n_groups,
//                                  Index_ n_rows,
//                                  Index_ max_rows,
//                                  Index_ stride,
//                                  Index_* row_starts,
//                                  Index_* row_steps,
//                                  T* data)
//     : n_groups(n_groups),
//       n_rows(n_rows),
//       max_rows(max_rows),
//       stride(stride),
//       row_starts(row_starts),
//       row_steps(row_steps),
//       m_data(data)
//   {
//   }

//   __device__ __forceinline__ T* dev_ptr(Index_ group_id) const
//   {
//     if (group_id >= n_groups) { return nullptr; }
//     return m_data + dev_row_starts[group_id] * stride;
//   }

//   __device__ __forceinline__ T* dev_ptr(Index_ group_id, Index_ group_rowid) const
//   {
//     if (blockIdx.x * blockDim.x + threadIdx.x + blockIdx.y * blockDim.y + threadIdx.y == 0) {
//       printf("dev_ptr in 1\n");
//       printf("dev_ptr in 11\n");
//       printf("n_groups = %ld\n", (long int)n_groups);
//       printf("dev_ptr in 111\n");
//       printf("n_groups = %ld\n", (long int)tmp);
//     }
//     if (group_id >= n_groups) { return nullptr; }
//     printf("dev_ptr in 2\n");
//     Index_ n_rows_group = dev_row_steps[group_id];
//     if (blockIdx.x * blockDim.x + threadIdx.x + blockIdx.y * blockDim.y + threadIdx.y == 0) {
//       printf("dev_ptr in 3\n");
//     }
//     if (group_rowid >= n_rows_group) { return nullptr; }
//     if (blockIdx.x * blockDim.x + threadIdx.x + blockIdx.y * blockDim.y + threadIdx.y == 0) {
//       printf("dev_ptr in 4\n");
//       printf("test host deref %ld\n", (long int)host_row_steps[group_id]);
//     }
//     return m_data + (dev_row_starts[group_id] + group_rowid) * stride;
//   }

//   __device__ __forceinline__ T* dev_ptr(Index_ group_id,
//                                         Index_ group_rowid,
//                                         Index_ group_colid) const
//   {
//     if (group_id >= n_groups || group_colid >= stride) { return nullptr; }
//     Index_ n_rows_group = dev_row_steps[group_id];
//     if (group_rowid >= n_rows_group) { return nullptr; }
//     return m_data + (dev_row_starts[group_id] + group_rowid) * stride + group_colid;
//   }

//   __host__ __forceinline__ T* host_ptr(Index_ group_id) const
//   {
//     if (group_id >= n_groups) { return nullptr; }
//     return m_data + host_row_starts[group_id] * stride;
//   }

//   __host__ __forceinline__ T* host_ptr(Index_ group_id, Index_ group_rowid) const
//   {
//     if (group_id >= n_groups) { return nullptr; }
//     Index_ n_rows_group = host_row_steps[group_id];
//     if (group_rowid >= n_rows_group) { return nullptr; }
//     return m_data + (host_row_starts[group_id] + group_rowid) * stride;
//   }

//   __host__ __forceinline__ T* host_ptr(Index_ group_id,
//                                        Index_ group_rowid,
//                                        Index_ group_colid) const
//   {
//     if (group_id >= n_groups || group_colid >= stride) { return nullptr; }
//     Index_ n_rows_group = host_row_steps[group_id];
//     if (group_rowid >= n_rows_group) { return nullptr; }
//     return m_data + (host_row_starts[group_id] + group_rowid) * stride + group_colid;
//   }

//   Index_ tmp;
//   Index_ n_groups;
//   Index_ n_rows;
//   Index_ max_rows;
//   Index_ stride;
//   // Index_* dev_row_starts  = nullptr;
//   // Index_* dev_row_steps   = nullptr;
//   // Index_* host_row_starts = nullptr;
//   // Index_* host_row_steps  = nullptr;
//   Index_* row_starts = nullptr;
//   Index_* row_steps  = nullptr;
//   T* m_data;
// };

}  // namespace Dbscan
}  // namespace ML