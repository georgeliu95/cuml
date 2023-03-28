/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.
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

#include <raft/core/handle.hpp>

#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>

#include "../multigroups/mg_accessor.cuh"

namespace ML {
namespace Dbscan {
namespace CorePoints {

/**
 * Compute the core points from the vertex degrees and min_pts criterion
 * @param[in]  handle          cuML handle
 * @param[in]  vd              Vertex degrees
 * @param[out] mask            Boolean core point mask
 * @param[in]  min_pts         Core point criterion
 * @param[in]  start_vertex_id First point of the batch
 * @param[in]  batch_size      Batch size
 * @param[in]  stream          CUDA stream
 */
template <typename Index_ = int>
void compute(const raft::handle_t& handle,
             const Index_* vd,
             bool* mask,
             Index_ min_pts,
             Index_ start_vertex_id,
             Index_ batch_size,
             cudaStream_t stream)
{
  auto counting = thrust::make_counting_iterator<Index_>(0);
  thrust::for_each(
    handle.get_thrust_policy(), counting, counting + batch_size, [=] __device__(Index_ idx) {
      mask[idx + start_vertex_id] = vd[idx] >= min_pts;
    });
}

template <typename Index_ = int>
void multi_group_compute(const raft::handle_t& handle,
                         const Metadata::VertexDegAccessor<Index_, Index_>& vd,
                         Metadata::CorePointAccessor<bool, Index_>& core_pts,
                         const Index_* min_pts)
{
  Index_ n_groups  = core_pts.nGroups;
  Index_ n_samples = core_pts.nSamples;
  const Index_ *vd_base = vd.vd;
  bool *mask = core_pts.data;
  const Index_ *offset = core_pts.offsetMask;

  auto counting = thrust::make_counting_iterator<Index_>(0);
  thrust::for_each(
    handle.get_thrust_policy(), counting, counting + n_samples, [=] __device__(Index_ idx) {
      mask[idx] = vd_base[idx] >= *(min_pts + offset[idx]);
    });
  return;
}

}  // namespace CorePoints
}  // namespace Dbscan
}  // namespace ML
