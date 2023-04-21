#pragma once

#include "mg_accessor.cuh"
#include "mg_csr.cuh"

namespace ML {
namespace Dbscan {
namespace Multigroups {
namespace AdjGraph {

template <typename Index_ = int>
void launcher(const raft::handle_t& handle,
              Metadata::AdjGraphAccessor<bool, Index_>& adj_ac,
              const Metadata::VertexDegAccessor<Index_, Index_>& vd_ac,
              Index_* adj_graph,
              Index_ adjnnz,
              Index_* ex_scan,
              Index_* row_counters,
              cudaStream_t stream)
{
  Index_* vd      = vd_ac.vd;
  Index_ n_points = vd_ac.n_points;

  // Compute the exclusive scan of the vertex degrees
  using namespace thrust;
  device_ptr<Index_> dev_vd      = device_pointer_cast(vd);
  device_ptr<Index_> dev_ex_scan = device_pointer_cast(ex_scan);
  thrust::exclusive_scan(handle.get_thrust_policy(), dev_vd, dev_vd + n_points, dev_ex_scan);

  Csr::multi_group_adj_to_csr(handle, adj_ac, ex_scan, row_counters, adj_graph, stream);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

template <typename Index_ = int>
void run(const raft::handle_t& handle,
         Metadata::AdjGraphAccessor<bool, Index_>& adj_ac,
         const Metadata::VertexDegAccessor<Index_, Index_>& vd_ac,
         Index_* adj_graph,
         Index_ adjnnz,
         Index_* ex_scan,
         int algo,
         Index_* row_counters,
         cudaStream_t stream)
{
  switch (algo) {
    case 0:
      ASSERT(
        false, "Incorrect algo '%d' passed! Naive version of adjgraph has been removed.", algo);
    case 1:
      launcher<Index_>(handle, adj_ac, vd_ac, adj_graph, adjnnz, ex_scan, row_counters, stream);
      break;
    default: ASSERT(false, "Incorrect algo passed! '%d'", algo);
  }
}

}  // namespace AdjGraph
}  // namespace Multigroups
}  // namespace Dbscan
}  // namespace ML
