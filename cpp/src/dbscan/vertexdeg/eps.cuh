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

#include <cuda_runtime.h>
#include <math.h>

#include <raft/device_utils.cuh>
#include <raft/linalg/contractions.cuh>

#include "pack.h"

template <typename DataT,
          typename IdxT,
          typename Policy,
          typename BaseClass = raft::linalg::Contractions_NT<DataT, IdxT, Policy>>
struct EpsUnexpL2SqNeighborhoodBatched : public BaseClass {
 private:
  typedef Policy P;

  IdxT lo;
  IdxT hi;

  bool* adj;
  DataT eps;
  IdxT* vd;

  char* smem;  // for final reductions

  DataT acc[P::AccRowsPerTh][P::AccColsPerTh];

 public:
  DI EpsUnexpL2SqNeighborhoodBatched(bool* _adj,
                              IdxT* _vd,
                              const DataT* _x,
                              const DataT* _y,
                              IdxT _m,
                              IdxT _n,
                              IdxT _k,
                              IdxT _lo,
                              IdxT _hi,
                              DataT _eps,
                              char* _smem)
    : BaseClass(_x, _y, _m, _n, _k, _smem), lo(_lo), hi(_hi), adj(_adj), eps(_eps), vd(_vd), smem(_smem)
  {
    // printf("m = %ld n = %ld\n", this->m, this->n);
  }

  DI void run()
  {
    // if(blockIdx.x * blockDim.x + threadIdx.x + blockIdx.y * blockDim.y + threadIdx.y == 0)
    // {
    //   printf("m = %ld, n = %ld, k = %ld\n", (long int)(this->m), (long int)(this->n), (long int)(this->k));
    // }
    
    prolog();
    loop();
    epilog();
  }

 private:
  DI void prolog()
  {
    // long long int start_time = 0;
    // if(blockIdx.x * blockDim.x + threadIdx.x + blockIdx.y * blockDim.y + threadIdx.y == 0)
    // {
    //   start_time = clock64();
    // }
    this->ldgXY(0);
#pragma unroll
    for (int i = 0; i < P::AccRowsPerTh; ++i) {
#pragma unroll
      for (int j = 0; j < P::AccColsPerTh; ++j) {
        acc[i][j] = BaseClass::Zero;
      }
    }
    this->stsXY();
    __syncthreads();
    this->pageWr ^= 1;
    // if(blockIdx.x * blockDim.x + threadIdx.x + blockIdx.y * blockDim.y + threadIdx.y == 0)
    // {
    //   long long int duration_time = clock64() - start_time;
    //   printf("prolog = %lld\n", duration_time);
    // }
  }

  DI void loop()
  {
    // long long int start_time = 0;
    // if(blockIdx.x * blockDim.x + threadIdx.x + blockIdx.y * blockDim.y + threadIdx.y == 0)
    // {
    //   start_time = clock64();
    // }
    for (int kidx = P::Kblk; kidx < this->k; kidx += P::Kblk) {
      this->ldgXY(kidx);
      accumulate();  // on the previous k-block
      this->stsXY();
      __syncthreads();
      this->pageWr ^= 1;
      this->pageRd ^= 1;
    }
    accumulate();  // last iteration
    // if(blockIdx.x * blockDim.x + threadIdx.x + blockIdx.y * blockDim.y + threadIdx.y == 0)
    // {
    //   long long int duration_time = clock64() - start_time;
    //   printf("loop = %lld\n", duration_time);
    // }
  }

  DI void epilog()
  {
    // long long int start_time = 0;
    // if(blockIdx.x * blockDim.x + threadIdx.x + blockIdx.y * blockDim.y + threadIdx.y == 0)
    // {
    //   start_time = clock64();
    // }
    IdxT startx = blockIdx.x * P::Mblk + this->accrowid;
    IdxT starty = blockIdx.y * P::Nblk + this->acccolid;
    auto lid    = raft::laneId();
    IdxT sums[P::AccColsPerTh];
#pragma unroll
    for (int j = 0; j < P::AccColsPerTh; ++j) {
      sums[j] = 0;
    }
#pragma unroll
    for (int i = 0; i < P::AccRowsPerTh; ++i) {
      auto xid = startx + i * P::AccThRows;
      if (xid < this->lo || xid >= this->hi) {
        continue; /* */
      }
#pragma unroll
      for (int j = 0; j < P::AccColsPerTh; ++j) {
        auto yid      = starty + j * P::AccThCols;
        auto is_neigh = acc[i][j] <= eps;
        ///@todo: fix uncoalesced writes using shared mem
        if (xid < this->m && yid < this->n) {
          adj[xid * this->n + yid] = is_neigh;
          sums[j] += is_neigh;
        }
      }
    }
    // perform reduction of adjacency values to compute vertex degrees
    if (vd != nullptr) { updateVertexDegree(sums); }
    // if(blockIdx.x * blockDim.x + threadIdx.x + blockIdx.y * blockDim.y + threadIdx.y == 0)
    // {
    //   long long int duration_time = clock64() - start_time;
    //   printf("epilog = %lld\n", duration_time);
    // }
  }

  DI void accumulate()
  {
    IdxT startx = blockIdx.x * P::Mblk + this->accrowid; 
#pragma unroll
    for (int ki = 0; ki < P::Kblk; ki += P::Veclen) {
      this->ldsXY(ki);
#pragma unroll
      for (int i = 0; i < P::AccRowsPerTh; ++i) {
        auto xid = startx + i * P::AccThRows;
        if (xid < this->lo || xid >= this->hi) {
          continue; /* */
        }
#pragma unroll
        for (int j = 0; j < P::AccColsPerTh; ++j) {
#pragma unroll
          for (int v = 0; v < P::Veclen; ++v) {
            auto diff = this->regx[i][v] - this->regy[j][v];
            acc[i][j] += diff * diff;
          }
        }
      }
    }
  }

  DI void updateVertexDegree(IdxT (&sums)[P::AccColsPerTh])
  {
    __syncthreads();  // so that we can safely reuse smem
    int gid       = threadIdx.x / P::AccThCols;
    int lid       = threadIdx.x % P::AccThCols;
    auto cidx     = IdxT(blockIdx.y) * P::Nblk + lid;
    IdxT totalSum = 0;
    // update the individual vertex degrees
#pragma unroll
    for (int i = 0; i < P::AccColsPerTh; ++i) {
      sums[i]  = raft::batchedBlockReduce<IdxT, P::AccThCols>(sums[i], smem);
      auto cid = cidx + i * P::AccThCols;
      if (gid == 0 && cid < this->n) {
        atomicUpdate(cid, sums[i]);
        totalSum += sums[i];
      }
      __syncthreads();  // for safe smem reuse
    }
    // update the total edge count
    totalSum = raft::blockReduce<IdxT>(totalSum, smem);
    if (threadIdx.x == 0) { atomicUpdate(this->n, totalSum); }
  }

  DI void atomicUpdate(IdxT addrId, IdxT val)
  {
    if (sizeof(IdxT) == 4) {
      raft::myAtomicAdd<unsigned>((unsigned*)(vd + addrId), val);
    } else if (sizeof(IdxT) == 8) {
      raft::myAtomicAdd<unsigned long long>((unsigned long long*)(vd + addrId), val);
    }
  }
};  // struct EpsUnexpL2SqNeighborhoodBatched

template <typename DataT, typename IdxT, typename Policy>
__global__ __launch_bounds__(Policy::Nthreads, 2)

static void epsUnexpL2SqNeighKernelBatched(
    bool* adj, IdxT* vd, const DataT* x, const DataT* y, IdxT m, IdxT n, IdxT k, IdxT lo, IdxT hi, DataT eps)
{
  extern __shared__ char smem[];
  EpsUnexpL2SqNeighborhoodBatched<DataT, IdxT, Policy> obj(adj, vd, x, y, m, n, k, lo, hi, eps, smem);
  obj.run();
}

template <typename DataT, typename IdxT, int VecLen>
static void epsUnexpL2SqNeighImplBatched(bool* adj,
                                  IdxT* vd,
                                  const DataT* x,
                                  const DataT* y,
                                  IdxT m,
                                  IdxT n,
                                  IdxT k,
                                  IdxT lo,
                                  IdxT hi,
                                  DataT eps,
                                  cudaStream_t stream)
{
  typedef typename raft::linalg::Policy4x4<DataT, VecLen>::Policy Policy;
  dim3 grid(raft::ceildiv<int>(m, Policy::Mblk), raft::ceildiv<int>(n, Policy::Nblk));
  dim3 blk(Policy::Nthreads);
  epsUnexpL2SqNeighKernelBatched<DataT, IdxT, Policy>
    <<<grid, blk, Policy::SmemSize, stream>>>(adj, vd, x, y, m, n, k, lo, hi, eps);
  RAFT_CUDA_TRY(cudaGetLastError());
}

template <typename DataT, typename IdxT>
void epsUnexpL2SqNeighborhoodBatched(bool* adj,
                                    IdxT* vd,
                                    const DataT* x,
                                    const DataT* y,
                                    IdxT m,
                                    IdxT n,
                                    IdxT k,
                                    IdxT lo,
                                    IdxT hi,
                                    DataT eps,
                                    cudaStream_t stream)
{
  size_t bytes = sizeof(DataT) * k;
  if (16 % sizeof(DataT) == 0 && bytes % 16 == 0) {
    epsUnexpL2SqNeighImplBatched<DataT, IdxT, 16 / sizeof(DataT)>(adj, vd, x, y, m, n, k, lo, hi, eps, stream);
  } else if (8 % sizeof(DataT) == 0 && bytes % 8 == 0) {
    epsUnexpL2SqNeighImplBatched<DataT, IdxT, 8 / sizeof(DataT)>(adj, vd, x, y, m, n, k, lo, hi, eps, stream);
  } else {
    epsUnexpL2SqNeighImplBatched<DataT, IdxT, 1>(adj, vd, x, y, m, n, k, lo, hi, eps, stream);
  }
}

