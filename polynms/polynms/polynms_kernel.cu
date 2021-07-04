
#include <thrust/sort.h>
#include <stdio.h>
#include "gpu_polynms.hpp"
#include "gpu_polygon.cuh"
#include <vector>
#include <iostream>

#define CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cudaError_t error = condition; \
    if (error != cudaSuccess) { \
      std::cout << cudaGetErrorString(error) << std::endl; \
    } \
  } while (0)

#define DIVUP(m,n) ((m) / (n) + ((m) % (n) > 0))
int const threadsPerBlock = sizeof(unsigned long long) * 8;
int const MAX_SIZE = 8 * 2;

__device__ float devIoU(float const * const a, float const * const b) {
    const int num_pts=4;
    float Sa = polygonArea(a, num_pts);
    float Sb = polygonArea(b, num_pts);
    int new_size;
    float p_inter[MAX_SIZE];
    float p_buffer[MAX_SIZE];
    suthHodgClip(a, num_pts, b, num_pts, p_inter, new_size, p_buffer);
    float interS = polygonArea(p_inter, new_size);
    //if(new_size!=4){
    //printf("%d, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f\n", new_size, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]);
    //}
    return interS / (Sa + Sb - interS);
}

__global__ void nms_kernel(const int n_boxes, const float nms_overlap_thresh,
                           const float *dev_boxes, unsigned long long *dev_mask){
  const int row_start = blockIdx.y;
  const int col_start = blockIdx.x;

  // if (row_start > col_start) return;

  const int row_size =
        min(n_boxes - row_start * threadsPerBlock, threadsPerBlock);
  const int col_size =
        min(n_boxes - col_start * threadsPerBlock, threadsPerBlock);

  __shared__ float block_boxes[threadsPerBlock * 8];
  if (threadIdx.x < col_size) {
    block_boxes[threadIdx.x * 8 + 0] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 8 + 0];
    block_boxes[threadIdx.x * 8 + 1] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 8 + 1];
    block_boxes[threadIdx.x * 8 + 2] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 8 + 2];
    block_boxes[threadIdx.x * 8 + 3] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 8 + 3];
    block_boxes[threadIdx.x * 8 + 4] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 8 + 4];
    block_boxes[threadIdx.x * 8 + 5] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 8 + 5];
    block_boxes[threadIdx.x * 8 + 6] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 8 + 6];
    block_boxes[threadIdx.x * 8 + 7] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 8 + 7];
  }

  __syncthreads();

  if (threadIdx.x < row_size) {
    const int cur_box_idx = threadsPerBlock * row_start + threadIdx.x;
    const float *cur_box = dev_boxes + cur_box_idx * 8;

    int i = 0;
    unsigned long long t = 0;
    int start = 0;
    if (row_start == col_start) {
      start = threadIdx.x + 1;
    }
    for (i = start; i < col_size; i++) {
      if (devIoU(cur_box, block_boxes + i * 8) > nms_overlap_thresh) {
        t |= 1ULL << i;
      }
    }
    const int col_blocks = DIVUP(n_boxes, threadsPerBlock);
    dev_mask[cur_box_idx * col_blocks + col_start] = t;
  }
}

void _set_device(int device_id) {
  int current_device;
  CUDA_CHECK(cudaGetDevice(&current_device));
  if (current_device == device_id) {
    return;
  }
  // The call to cudaSetDevice must come before any calls to Get, which
  // may perform initialization using the GPU.
  CUDA_CHECK(cudaSetDevice(device_id));
}

void _polynms(int* keep_out, int* num_out, const float* boxes_host, int boxes_num,
          int boxes_dim, float nms_overlap_thresh, int device_id) {
  _set_device(device_id);

  float* boxes_dev = NULL;
  unsigned long long* mask_dev = NULL;

  const int col_blocks = DIVUP(boxes_num, threadsPerBlock);

  CUDA_CHECK(cudaMalloc(&boxes_dev,
                        boxes_num * boxes_dim * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(boxes_dev,
                        boxes_host,
                        boxes_num * boxes_dim * sizeof(float),
                        cudaMemcpyHostToDevice));

  CUDA_CHECK(cudaMalloc(&mask_dev,
                        boxes_num * col_blocks * sizeof(unsigned long long)));

  dim3 blocks(DIVUP(boxes_num, threadsPerBlock),
              DIVUP(boxes_num, threadsPerBlock));
  dim3 threads(threadsPerBlock);
  nms_kernel<<<blocks, threads>>>(boxes_num,
                                  nms_overlap_thresh,
                                  boxes_dev,
                                  mask_dev);

  std::vector<unsigned long long> mask_host(boxes_num * col_blocks);
  CUDA_CHECK(cudaMemcpy(&mask_host[0],
                        mask_dev,
                        sizeof(unsigned long long) * boxes_num * col_blocks,
                        cudaMemcpyDeviceToHost));

  std::vector<unsigned long long> remv(col_blocks);
  memset(&remv[0], 0, sizeof(unsigned long long) * col_blocks);

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
  *num_out = num_to_keep;

  CUDA_CHECK(cudaFree(boxes_dev));
  CUDA_CHECK(cudaFree(mask_dev));
}
