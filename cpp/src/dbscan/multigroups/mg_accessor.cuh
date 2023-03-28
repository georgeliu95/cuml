#pragma once

#include <cstdio>
#include <cstdlib>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>

#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>
#include <raft/core/handle.hpp>

#include <rmm/mr/host/host_memory_resource.hpp>


namespace ML {
namespace Dbscan {
namespace Metadata {

template <typename T>
struct multiply_scalar
{
  multiply_scalar(T _scalar) : scalar(_scalar) {}
  __host__ __device__
  T operator()(T value) const
  {
    return scalar * value;
  }
  T scalar;
};

template <typename T>
struct sq
{
  __host__ __device__
  T operator()(T value) const
  {
    return value * value;
  }
};

template <typename Index_ = int>
__global__ void initOffsetMask(Index_* mask, Index_* stride, Index_* position, int nGroups) {
  int groupIdx = blockIdx.x;
  int rowIdx = threadIdx.x;
  if(groupIdx >= nGroups)
    return;
  Index_* offsetMaskBase = mask + position[groupIdx];
  while(rowIdx < stride[groupIdx]) {
    offsetMaskBase[rowIdx] = groupIdx;
    rowIdx += blockDim.x;
  }
  return;
}

const std::size_t align = 256;

template <typename Index_t = int>
class MultiGroupMetaData {
 public:
  MultiGroupMetaData(Index_t inNbGroups, const Index_t *inNbRows, Index_t inNbCols)
    : nGroups(inNbGroups), 
      nRows(inNbRows),
      isConstCols(true) {
    this->nCols = reinterpret_cast<Index_t*>(
      rmm::mr::host_memory_resource::allocate(nGroups * sizeof(Index_t)));
    thrust::fill_n(thrust::host, this->nCols, nGroups, inNbCols);
    
    this->nRowsMax = *thrust::max_element(thrust::host, inNbRows, inNbRows + inNbGroups);
    this->nColsMax = inNbCols;
    this->nRowsSum = thrust::reduce(thrust::host, inNbRows, inNbRows + inNbGroups);
    this->nColsSum = inNbCols * inNbGroups;
  }

  MultiGroupMetaData(Index_t inNbGroups, const Index_t *inNbRows, const Index_t *inNbCols)
    : nGroups(inNbGroups), 
      nRows(inNbRows),
      nCols(inNbCols),
      isConstCols(false) {
    this->nRowsMax = *thrust::max_element(thrust::host, inNbRows, inNbRows + inNbGroups);
    this->nColsMax = *thrust::max_element(thrust::host, inNbCols, inNbCols + inNbGroups);
    this->nRowsSum = thrust::reduce(thrust::host, inNbRows, inNbRows + inNbGroups);
    this->nColsSum = thrust::reduce(thrust::host, inNbCols, inNbCols + inNbGroups);
  }

  ~MultiGroupMetaData() {
    if(this->isConstCols) {
      rmm::mr::host_memory_resource::deallocate(this->nCols, nGroups * sizeof(Index_t));
    }
  }

  size_t getWorkspaceSize() {
    size_t workspaceSize = raft::alignTo<std::size_t>(sizeof(Index_t) * (this->nGroups + 1), align);
    workspaceSize *= (this->isConstCols)? 2 : 4;
    return workspaceSize;
  }

  void initialize(const raft::handle_t& handle, void *workspace, size_t bufferSize, cudaStream_t stream) {
    this->workspaceSize = this->getWorkspaceSize();
    ASSERT(bufferSize == this->workspaceSize,
      "The required size of workspace (%ld) doesn't match that passed (%ld) in %s.\n",
      this->workspaceSize, bufferSize, __FUNCTION__);
    size_t chunkSize = this->workspaceSize / ((this->isConstCols)? 2 : 4);
    Index_t *ptrWorkspace = reinterpret_cast<Index_t*>(workspace);
    this->_workspace = ptrWorkspace;
    this->reset(stream);

    this->devNbRows = ptrWorkspace;
    ptrWorkspace += chunkSize;
    this->prefixSumRows = ptrWorkspace;
    ptrWorkspace += chunkSize;
    RAFT_CUDA_TRY(cudaMemcpyAsync(
      this->devNbRows, 
      this->metadata.getPtrNbRows(), 
      this->metadata.nGroups * sizeof(Index_t), 
      cudaMemcpyHostToDevice, 
      stream
    ));
    thrust::device_ptr<Index_t> devArray = thrust::device_pointer_cast(this->devNbRows);
    thrust::device_ptr<Index_t> devPrefixSum = thrust::device_pointer_cast(this->devPrefixSumRows);
    thrust::exclusive_scan(handle.get_thrust_policy(), devArray, devArray + (this->metadata.nGroups + 1), devPrefixSum);

    if(this->metadata.isConstCols) {
      this->devNbCols = ptrWorkspace;
      ptrWorkspace += chunkSize;
      this->prefixSumCols = ptrWorkspace;
      thrust::device_ptr<Index_t> devArray = thrust::device_pointer_cast(this->devNbCols);
      thrust::device_ptr<Index_t> devPrefixSum = thrust::device_pointer_cast(this->devPrefixSumCols);
      thrust::exclusive_scan(handle.get_thrust_policy(), devArray, devArray + (this->metadata.nGroups + 1), devPrefixSum);
    }
    return;
  }

  void destroy() {
    this->workspaceSize = 0;
    this->_workspace = nullptr;
    this->devNbRows = nullptr;
    this->devNbCols = nullptr;
    this->devPrefixSumRows = nullptr;
    this->devPrefixSumCols = nullptr;
    return;
  }

  HDI const Index_t* getHostNbRows() const noexcept { return this->nRows; }
  HDI const Index_t* getHostNbCols() const noexcept { return this->nCols; }
  HDI const Index_t* getDevNbRows() const noexcept { return this->devNbRows; }
  HDI const Index_t* getDevNbCols() const noexcept { return this->devNbCols; }
  HDI const Index_t* getDevPrefixSumRows() const noexcept { return this->devPrefixSumRows; }
  HDI const Index_t* getDevPrefixSumCols() const noexcept { return this->devPrefixSumCols; }

  const Index_t nGroups;
  Index_t nRowsMax;
  Index_t nRowsSum;
  Index_t nColsMax;
  Index_t nColsSum;
  const bool isConstCols;
 private:
  void reset(cudaStream_t stream) {
    if(this->_workspace != nullptr) {
      RAFT_CUDA_TRY(cudaMemsetAsync(this->_workspace, 0, this->workspaceSize, stream));
    }
  }

  const Index_t *nRows = nullptr;
  const Index_t *nCols = nullptr;
  Index_t *devNbRows = nullptr;
  Index_t *devNbCols = nullptr;
  Index_t *devPrefixSumRows = nullptr;
  Index_t *devPrefixSumCols = nullptr;
  size_t workspaceSize = 0;
  Index_t *_workspace = nullptr;
  
};

template <typename Data_t,
          typename Index_t = int,
          typename MetaDataClass = MultiGroupMetaData<Index_t>>
class BaseAccessor {
 public:
  BaseAccessor(const MetaDataClass& inMetadata, Data_t *inData)
   : metadata(inMetadata), 
     nGroups(inMetadata.nGroups), 
     nRows(inMetadata.getDevNbRows()),
     startRows(inMetadata.getDevPrefixSumRows()),
     readonly_data(inData),
     data(inData) {}
  BaseAccessor(const MetaDataClass& inMetadata, const Data_t *inData)
   : metadata(inMetadata), 
     nGroups(inMetadata.nGroups), 
     nRows(inMetadata.getDevNbRows()),
     startRows(inMetadata.getDevPrefixSumRows()),
     readonly_data(inData) {}

  virtual void initialize(const raft::handle_t& handle, void *workspace, size_t bufferSize, cudaStream_t stream) {}
  virtual void destroy() {}

  const MetaDataClass metadata;
  const Index_t nGroups;
  const Index_t *nRows;
  const Index_t *startRows;
  Data_t *data = nullptr;
  const Data_t *readonly_data = nullptr;
};

template <typename Data_t,
          typename Index_t = int,
          typename MetaDataClass = MultiGroupMetaData<Index_t>,
          typename BaseClass = BaseAccessor<Data_t, Index_t, MetaDataClass>>
class PointAccessor : public BaseClass {
 public:
  PointAccessor(const MetaDataClass& inMetadata, const Data_t *inData) 
   : BaseClass(inMetadata, inData),
     nRowsSum(inMetadata.nRowsSum),
     nRowsMax(inMetadata.nRowsMax),
     stride(inMetadata.nColsMax) {}

  const Index_t nRowsSum;
  const Index_t nRowsMax;
  const Index_t stride;
};

template <typename Data_t,
          typename Index_t = int,
          typename MetaDataClass = MultiGroupMetaData<Index_t>,
          typename BaseClass = BaseAccessor<Data_t, Index_t, MetaDataClass>>
class VertexDegAccessor : public BaseClass {
 public:
  VertexDegAccessor(const MetaDataClass& inMetadata, Data_t *inData) 
   : BaseClass(inMetadata, inData), nPoints(inMetadata.nRowsSum) {
    Data_t temp = inData;
    this->vd = temp;
    temp += inMetadata.nRowsSum;
    this->vd_all = temp;
    temp += 1;
    this->vd_group = temp;
  }

  const Index_t nPoints;
  Data_t *vd;
  Data_t *vd_group;
  Data_t *vd_all;
};

template <typename Data_t,
          typename Index_t = int,
          typename MetaDataClass = MultiGroupMetaData<Index_t>,
          typename BaseClass = BaseAccessor<Data_t, Index_t, MetaDataClass>>
class AdjGraphAccessor : public BaseClass {
 public:
  AdjGraphAccessor(const MetaDataClass& inMetadata, Data_t *inData) : BaseClass(inMetadata, inData),
    stride(inMetadata.nRowsMax),
    nRows(inMetadata.getDevNbRows()),
    startRows(inMetadata.getDevPrefixSumRows()) {}

  size_t getWorkspaceSize() { 
    return 2 * raft::alignTo<std::size_t>(this->nGroups * sizeof(Index_t));
  }

  size_t getLayoutSize() {
    return raft::alignTo<std::size_t>(sizeof(bool) * this->metadata.nRowsSum * this->stride, align);
  }

  void initialize(const raft::handle_t& handle, void *workspace, size_t bufferSize, cudaStream_t stream) {
    this->workspaceSize = this->getWorkspaceSize();
    ASSERT(bufferSize == this->workspaceSize,
      "The required size of workspace (%ld) doesn't match that passed (%ld) in %s.\n",
      this->workspaceSize, bufferSize, __FUNCTION__);
    
    if(this->adjStride != nullptr || this->groupOffset != nullptr) {
      this->destroy();
    }
    this->adjStride = reinterpret_cast<Index_t*>(workspace);
    this->groupOffset = reinterpret_cast<Index_t*>(workspace) + bufferSize / 2;
    RAFT_CUDA_TRY(
      cudaMemcpy(
        this->adjStride, this->nRows, this->nGroups * sizeof(Index_t), cudaMemcpyDeviceToDevice, stream));
    RAFT_CUDA_TRY(
      cudaMemcpy(
        this->groupOffset, this->startRows, this->nGroups * sizeof(Index_t), cudaMemcpyDeviceToDevice, stream));

    thrust::device_ptr<Index_t> dev_offset = thrust::device_pointer_cast(this->groupOffset);
    thrust::for_each(
      handle.get_thrust_policy(), dev_offset, dev_offset + this->nGroups, multiply_scalar<Index_t>(this->stride));
    return;
  }
  void destroy() {
    this->adjStride = nullptr;
    this->groupOffset = nullptr;
    return;
  }

  const Index_t stride;
  const Index_t *nRows;
  const Index_t *startRows;
  Index_t *adjStride = nullptr;
  Index_t *groupOffset = nullptr;
  size_t workspaceSize;
};

template <typename Data_t,
          typename Index_t = int,
          typename MetaDataClass = MultiGroupMetaData<Index_t>,
          typename BaseClass = BaseAccessor<Data_t, Index_t, MetaDataClass>>
class CorePointAccessor : public BaseClass {
 public:
  CorePointAccessor(const MetaDataClass& inMetadata, Data_t *inData) 
   : BaseClass(inMetadata, inData), nSamples(inMetadata.nRowsSum) {}

  size_t getWorkspaceSize() { 
    return raft::alignTo<std::size_t>(this->nGroups * sizeof(Index_t));
  }

  void initialize(const raft::handle_t& handle, void *workspace, size_t bufferSize, cudaStream_t stream) {
    this->workspaceSize = this->getWorkspaceSize();
    ASSERT(bufferSize == this->workspaceSize,
      "The required size of workspace (%ld) doesn't match that passed (%ld) in %s.\n",
      this->workspaceSize, bufferSize, __FUNCTION__);
    
    if(this->offsetMask != nullptr) {
      this->destroy();
    }
    this->offsetMask = reinterpret_cast<Index_t*>(workspace);
    dim3 gridSize {this->nGroups};
    dim3 blkSize {align / sizeof(Index_t)};
    initOffsetMask<<<gridSize, blkSize, 0, stream>>>(this->offsetMask, this->nRows, this->startRows, this->nGroups);
    RAFT_CUDA_TRY(cudaPeekAtLastError());
  }

  void destroy() {
    this->offsetMask = nullptr;
  }

  Index_t nSamples;
  Index_t *offsetMask = nullptr;
  size_t workspaceSize;
};


// template <typename Data_t,
//           typename Index_t = int,
//           typename MetaDataClass = MultiGroupMetaData<Index_t>,
//           typename BaseClass = BaseAccessor<Data_t, Index_t, MetaDataClass>>
// class PointAccessorDeprecated : public BaseClass {
//  public:
//   PointAccessorDeprecated(const MetaDataClass& inMetadata, const Data_t *inData)
//    : BaseClass(inMetadata, inData) {
//     this->nLgRows = inMetadata.nRowsMax;
//     this->nLgCols = inMetadata.nColsMax;
//     this->stride = inMetadata.nRowsMax * inMetadata.nColsMax;
//   }

//   /* 
//   HDI Data_t& operator[] (Index_t idx) {
//     Index_t groupId = idx / stride;
//     Index_t rowId = (idx / nLgCols) % nLgRows;
//     Index_t colId = idx % nLgCols;
//     if(rowId < this->nRowsPerGroup[groupId]) {
//       return this->readonly_data[groupBase[groupId] + rowId * nLgCols + colId];
//     } else {
//       return static_cast<Data_t>(0);
//     }
//   } 
//   */

//   Index_t nLgRows;
//   Index_t nLgCols;
//   Index_t stride;
//  private:
//   Index_t *nRowsPerGroup;
//   Index_t *groupBase;
// };

}  // namespace Metadata
}  // namespace Dbscan
}  // namespace ML
