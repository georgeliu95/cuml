#pragma once

#include "mg_accessor.cuh"

namespace ML {
namespace Dbscan {
namespace Multigroups {
namespace CorePoints {

template <typename Index_ = int>
void multi_group_compute(const raft::handle_t& handle,
                         const Metadata::VertexDegAccessor<Index_, Index_>& vd_ac,
                         Metadata::CorePointAccessor<bool, Index_>& corepts_ac,
                         const Index_* min_pts)
{
  Index_ n_groups       = corepts_ac.n_groups;
  Index_ n_samples      = corepts_ac.n_points;
  const Index_* vd_base = vd_ac.vd;
  bool* mask            = corepts_ac.core_pts;
  const Index_* offset  = corepts_ac.offset_mask;

  auto counting = thrust::make_counting_iterator<Index_>(0);
  thrust::for_each(
    handle.get_thrust_policy(), counting, counting + n_samples, [=] __device__(Index_ idx) {
      mask[idx] = vd_base[idx] >= *(min_pts + offset[idx]);
    });
  return;
}

}  // namespace CorePoints
}  // namespace Multigroups
}  // namespace Dbscan
}  // namespace ML
