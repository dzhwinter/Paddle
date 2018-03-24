//   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <cuda_runtime.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include "paddle/fluid/platform/cuda_helper.h"
using namespace paddle::platform;

// function unittest and compare the speed difference
// between two kernel implementation.
// usage: nvprof ./a.out
template <typename T>
__global__ void naive_add_kernel(T* data, size_t n) {
  extern __shared__ T shm[];
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid >= n) return;
  shm[tid] = data[tid];
  // force sync to enforce write to shm
  __syncthreads();
  for (int offset = n / 2; offset > 0; offset /= 2) {
    if (tid < offset) {
      CudaAtomicAdd(&data[tid], data[tid + offset]);
    }
  }
}

template <typename T>
__global__ void reduction_kernel(T* data, size_t n) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  T val = data[tid];
  reduceSum<T>(val, tid, n);
}

TEST(Reduction, Normal) {
  constexpr size_t size = 64;
  float *da = nullptr, *db = nullptr;
  // cudaMallocHost can be access in device and host
  ASSERT_EQ(cudaMallocHost((void**)&da, size * sizeof(float)), 0);
  ASSERT_EQ(cudaMallocHost((void**)&db, size * sizeof(float)), 0);
  ASSERT_EQ(cudaMemset(da, 1, size * sizeof(float)), 0);
  ASSERT_EQ(cudaMemset(db, 1, size * sizeof(float)), 0);
  reduction_kernel<float><<<32, 1024>>>(da, size);
  cudaDeviceSynchronize();
  naive_add_kernel<float><<<32, 1024, sizeof(float) * size>>>(db, size);
  cudaDeviceSynchronize();
  ASSERT_EQ(da[0], db[0]);
  VLOG(0) << da[0] << " " << db[0];

  cudaFree(da);
  cudaFree(db);
}
