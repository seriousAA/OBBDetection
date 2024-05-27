// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/DeviceGuard.h>

#include <vector>
#include <iostream>

#include <torch/extension.h>
// Derive major and minor version if not already defined
#ifndef TORCH_VERSION_MAJOR
#define TORCH_VERSION_MAJOR (TORCH_VERSION / 10000)
#endif

#ifndef TORCH_VERSION_MINOR
#define TORCH_VERSION_MINOR (TORCH_VERSION / 100 % 100)
#endif

#if TORCH_VERSION_MAJOR < 1 || (TORCH_VERSION_MAJOR == 1 && TORCH_VERSION_MINOR <= 10)
#include <THC/THC.h>
#include <THC/THCDeviceUtils.cuh>
#define CEIL_DIV(x, y) THCCeilDiv(x, y)
#define CUDA_MALLOC(size) THCudaMalloc(at::globalContext().getTHCState(), size)
#define CUDA_FREE(ptr) THCudaFree(at::globalContext().getTHCState(), ptr)
#define CUDA_CHECK(expr) THCudaCheck(expr)
#else
#include "ATen/cuda/DeviceUtils.cuh"
#include <c10/cuda/CUDACachingAllocator.h>
#include <ATen/ceil_div.h>
#define CEIL_DIV(x, y) at::ceil_div(x, y)
#define CUDA_MALLOC(size) c10::cuda::CUDACachingAllocator::raw_alloc(size)
#define CUDA_FREE(ptr) c10::cuda::CUDACachingAllocator::raw_delete(ptr)
#define CUDA_CHECK(expr) C10_CUDA_CHECK(expr)
#endif

int const threadsPerBlock = sizeof(unsigned long long) * 8;

__device__ inline float devIoU(float const * const a, float const * const b) {
  float left = max(a[0], b[0]), right = min(a[2], b[2]);
  float top = max(a[1], b[1]), bottom = min(a[3], b[3]);
  float width = max(right - left, 0.f), height = max(bottom - top, 0.f);
  float interS = width * height;
  float Sa = (a[2] - a[0]) * (a[3] - a[1]);
  float Sb = (b[2] - b[0]) * (b[3] - b[1]);
  return interS / (Sa + Sb - interS);
}

__global__ void nms_kernel(const int n_boxes, const float nms_overlap_thresh,
                           const float *dev_boxes, unsigned long long *dev_mask) {
  const int row_start = blockIdx.y;
  const int col_start = blockIdx.x;

  // if (row_start > col_start) return;

  const int row_size =
        min(n_boxes - row_start * threadsPerBlock, threadsPerBlock);
  const int col_size =
        min(n_boxes - col_start * threadsPerBlock, threadsPerBlock);

  __shared__ float block_boxes[threadsPerBlock * 5];
  if (threadIdx.x < col_size) {
    block_boxes[threadIdx.x * 5 + 0] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 5 + 0];
    block_boxes[threadIdx.x * 5 + 1] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 5 + 1];
    block_boxes[threadIdx.x * 5 + 2] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 5 + 2];
    block_boxes[threadIdx.x * 5 + 3] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 5 + 3];
    block_boxes[threadIdx.x * 5 + 4] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 5 + 4];
  }
  __syncthreads();

  if (threadIdx.x < row_size) {
    const int cur_box_idx = threadsPerBlock * row_start + threadIdx.x;
    const float *cur_box = dev_boxes + cur_box_idx * 5;
    int i = 0;
    unsigned long long t = 0;
    int start = 0;
    if (row_start == col_start) {
      start = threadIdx.x + 1;
    }
    for (i = start; i < col_size; i++) {
      if (devIoU(cur_box, block_boxes + i * 5) > nms_overlap_thresh) {
        t |= 1ULL << i;
      }
    }
    const int col_blocks = CEIL_DIV(n_boxes, threadsPerBlock);
    dev_mask[cur_box_idx * col_blocks + col_start] = t;
  }
}

// boxes is a N x 5 tensor
at::Tensor nms_cuda_forward(const at::Tensor boxes, float nms_overlap_thresh) {

  // Ensure CUDA uses the input tensor device.
  at::DeviceGuard guard(boxes.device());

  using scalar_t = float;
  AT_ASSERTM(boxes.device().is_cuda(), "boxes must be a CUDA tensor");
  auto scores = boxes.select(1, 4);
  auto order_t = std::get<1>(scores.sort(0, /* descending=*/true));
  auto boxes_sorted = boxes.index_select(0, order_t);

  int boxes_num = boxes.size(0);

  const int col_blocks = CEIL_DIV(boxes_num, threadsPerBlock);

  scalar_t* boxes_dev = boxes_sorted.data_ptr<scalar_t>();

  unsigned long long* mask_dev = NULL;

  mask_dev = (unsigned long long*) CUDA_MALLOC(boxes_num * col_blocks * sizeof(unsigned long long));

  dim3 blocks(CEIL_DIV(boxes_num, threadsPerBlock),
              CEIL_DIV(boxes_num, threadsPerBlock));
  dim3 threads(threadsPerBlock);
  nms_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(boxes_num,
                                  nms_overlap_thresh,
                                  boxes_dev,
                                  mask_dev);

  std::vector<unsigned long long> mask_host(boxes_num * col_blocks);
  CUDA_CHECK(cudaMemcpyAsync(
			  &mask_host[0],
			  mask_dev,
			  sizeof(unsigned long long) * boxes_num * col_blocks,
			  cudaMemcpyDeviceToHost,
			  at::cuda::getCurrentCUDAStream()
			  ));

  std::vector<unsigned long long> remv(col_blocks);
  memset(&remv[0], 0, sizeof(unsigned long long) * col_blocks);

  at::Tensor keep = at::empty({boxes_num}, boxes.options().dtype(at::kLong).device(at::kCPU));
  int64_t* keep_out = keep.data_ptr<int64_t>();

  int num_to_keep = 0;
  for (int i = 0; i < boxes_num; i++) {
    int nblock = i / threadsPerBlock;
    int inblock = i % threadsPerBlock;

    if (!(remv[nblock] & (1ULL << inblock))) {
      keep_out[num_to_keep++] = i;
      unsigned long long *p = &mask_host[0] + i * col_blocks;
      for (int j = nblock; j < col_blocks; j++) {
        remv[j] |= p[j];
      }
    }
  }

  CUDA_FREE(mask_dev);
  // TODO improve this part
  return order_t.index({
      keep.narrow(/*dim=*/0, /*start=*/0, /*length=*/num_to_keep).to(
          order_t.device(), keep.scalar_type())});
}
