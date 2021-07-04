
#include <stdio.h>
#include <time.h>
#include "gpu_iou_matrix.hpp"
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

#define CUDA_KERNEL_LOOP(i, n) \
   for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
        i < (n); \
        i += blockDim.x * gridDim.x)

// CUDA: use 512 threads per block
const int CAFFE_CUDA_NUM_THREADS = 512;

// CUDA: number of blocks for threads.
inline int CAFFE_GET_BLOCKS(const int N) {
  return (N + CAFFE_CUDA_NUM_THREADS - 1) / CAFFE_CUDA_NUM_THREADS;
}

int const MAX_SIZE = 8 * 2;

// Intersection-over-Union
__device__ float devIoU(float const * const a, float const * const b) {
    const int num_pts=4;
    float Sa = polygonArea(a, num_pts);
    float Sb = polygonArea(b, num_pts);
    int new_size;
    float p_inter[MAX_SIZE];
    float p_buffer[MAX_SIZE];
    suthHodgClip(a, num_pts, b, num_pts, p_inter, new_size, p_buffer);
    float interS = polygonArea(p_inter, new_size);
    return interS / (Sa + Sb - interS);
}

// Intersection-over-Area
__device__ float devIoA(float const * const a, float const * const b) {
    const int num_pts=4;
    float Sa = polygonArea(a, num_pts);
    float Sb = polygonArea(b, num_pts);
    int new_size;
    float p_inter[MAX_SIZE];
    float p_buffer[MAX_SIZE];
    suthHodgClip(a, num_pts, b, num_pts, p_inter, new_size, p_buffer);
    float interS = polygonArea(p_inter, new_size);
    return interS / Sb;
}

__global__ void ioa_kernel(const float *dev_boxes1, int num_boxes1, 
                           const float *dev_boxes2, int num_boxes2,
                           float *dev_matrix){
  CUDA_KERNEL_LOOP(i, num_boxes1*num_boxes2){
    const int row_id= i / num_boxes2;
    const int col_id= i % num_boxes2;
    const float* box1 = dev_boxes1 + row_id * 8;
    const float* box2 = dev_boxes2 + col_id * 8;
    float *cur_ioa = dev_matrix + row_id * num_boxes2 + col_id;
    *cur_ioa = devIoA(box1, box2);
  }
}

__global__ void iou_kernel(const float *dev_boxes1, int num_boxes1, 
                           const float *dev_boxes2, int num_boxes2,
                           float *dev_matrix){
  CUDA_KERNEL_LOOP(i, num_boxes1*num_boxes2){
    const int row_id= i / num_boxes2;
    const int col_id= i % num_boxes2;
    const float* box1 = dev_boxes1 + row_id * 8;
    const float* box2 = dev_boxes2 + col_id * 8;
    float *cur_iou = dev_matrix + row_id * num_boxes2 + col_id;
    *cur_iou = devIoU(box1, box2);
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

void _matrix_iou(float* matrix_iou, const float* boxes1_host, int boxes1_num, const float* boxes2_host, int boxes2_num,
          int boxes_dim, int device_id) {
  _set_device(device_id);

  float* boxes1_dev = NULL;
  float* boxes2_dev = NULL;
  float* mask_dev = NULL;

  CUDA_CHECK(cudaMalloc(&boxes1_dev,
                        boxes1_num * boxes_dim * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(boxes1_dev,
                        boxes1_host,
                        boxes1_num * boxes_dim * sizeof(float),
                        cudaMemcpyHostToDevice));

  CUDA_CHECK(cudaMalloc(&boxes2_dev,
                        boxes2_num * boxes_dim * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(boxes2_dev,
                        boxes2_host,
                        boxes2_num * boxes_dim * sizeof(float),
                        cudaMemcpyHostToDevice));

  CUDA_CHECK(cudaMalloc(&mask_dev,
                        boxes1_num * boxes2_num * sizeof(float)));

  //cudaEvent_t start, stop;
  //float elapsedTime;

  //cudaEventCreate(&start);
  //cudaEventRecord(start,0);
 //Do kernel activity here
  iou_kernel<<<CAFFE_GET_BLOCKS(boxes1_num*boxes2_num), CAFFE_CUDA_NUM_THREADS>>>(
                                  boxes1_dev, boxes1_num,
                                  boxes2_dev, boxes2_num,
                                  mask_dev);
  CUDA_CHECK(cudaMemcpy(matrix_iou,
                         mask_dev,
                         sizeof(float) * boxes1_num * boxes2_num,
                         cudaMemcpyDeviceToHost));

  //cudaEventCreate(&stop);
  //cudaEventRecord(stop,0);
  //cudaEventSynchronize(stop);

  //cudaEventElapsedTime(&elapsedTime, start,stop);
  //printf("Elapsed time : %f ms\n" ,elapsedTime);
 
  CUDA_CHECK(cudaFree(boxes1_dev));
  CUDA_CHECK(cudaFree(boxes2_dev));
  CUDA_CHECK(cudaFree(mask_dev));
}
