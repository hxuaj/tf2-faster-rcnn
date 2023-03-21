#include "gpu_nms.hpp"
#include <vector>
#include <iostream>

#define CHECK(call) \
    do \
    { \
        const cudaError_t error_code = call; \
        if (error_code != cudaSuccess) \
        { \
            std::cerr << "CUDA Error:" << std::endl; \
            std::cerr << "    File:       " << __FILE__ << std::endl; \
            std::cerr << "    Line:       " << __LINE__ << std::endl; \
            std::cerr << "    Error code: " << error_code << std::endl; \
            std::cerr << "    Error text: " << cudaGetErrorString(error_code) << std::endl; \
            exit(1); \
        } \
    } while (0)

void _set_device(int device_id)
{
    int cur_device_id;
    CHECK(cudaGetDevice(&cur_device_id));
    if (cur_device_id == device_id)
    {
        return;
    }
    CHECK(cudaSetDevice(device_id));
}

#define DTYPE uint32_t
#define DIVUP(m,n) ((m) / (n) + ((m) % (n) > 0))
const int TILE_DIM = sizeof(DTYPE) * 8; // TILE_DIM * TILE_DIM threads per block

// float const * const: const pointer to const float
__device__ inline float devIoU(float const * const a, float const * const b)
{
  float left = max(a[0], b[0]), right = min(a[2], b[2]);
  float top = max(a[1], b[1]), bottom = min(a[3], b[3]);
  float width = max(right - left + 1, 0.f), height = max(bottom - top + 1, 0.f);
  float interS = width * height;
  float Sa = (a[2] - a[0] + 1) * (a[3] - a[1] + 1);
  float Sb = (b[2] - b[0] + 1) * (b[3] - b[1] + 1);
  return interS / (Sa + Sb - interS);
}

__global__ void nms_kernel(const float *boxes_dev, DTYPE *mask_dev, const int n, const float thresh)
{
    // Calculate the block size
    const int row_size = min(n - blockIdx.x * TILE_DIM, TILE_DIM);
    const int col_size = min(n - blockIdx.y * TILE_DIM, TILE_DIM);
    const int row_idx = blockIdx.x * TILE_DIM + threadIdx.x; // x index map to boxes_dev
    const int col_idx = blockIdx.y * TILE_DIM + threadIdx.y; // y index map to boxes_dev

    __shared__ float block_boxes_x[TILE_DIM * 4]; // extra 4 bytes for avoiding bank conflicts
    __shared__ float block_boxes_y[TILE_DIM * 4];
    __shared__ DTYPE block_mask[TILE_DIM];

    if (threadIdx.x < row_size)
    {
        block_boxes_x[threadIdx.x * 4 + 0] = boxes_dev[row_idx * 4 + 0];
        block_boxes_x[threadIdx.x * 4 + 1] = boxes_dev[row_idx * 4 + 1];
        block_boxes_x[threadIdx.x * 4 + 2] = boxes_dev[row_idx * 4 + 2];
        block_boxes_x[threadIdx.x * 4 + 3] = boxes_dev[row_idx * 4 + 3];
    }
    if (threadIdx.y < col_size)
    {
        block_boxes_y[threadIdx.y * 4 + 0] = boxes_dev[col_idx * 4 + 0];
        block_boxes_y[threadIdx.y * 4 + 1] = boxes_dev[col_idx * 4 + 1];
        block_boxes_y[threadIdx.y * 4 + 2] = boxes_dev[col_idx * 4 + 2];
        block_boxes_y[threadIdx.y * 4 + 3] = boxes_dev[col_idx * 4 + 3];
        block_mask[threadIdx.y] = 0U;
    }
    __syncthreads();

    // IoU calculation
    if ((threadIdx.x < row_size) && (threadIdx.y < col_size) && (row_idx > col_idx))
    {
        if (devIoU(block_boxes_x + threadIdx.x * 4, block_boxes_y + threadIdx.y * 4) > thresh)
        {
            atomicOr(&block_mask[threadIdx.y], 1UL << threadIdx.x);
        }
    }
    // copy back to global memory
    if ((threadIdx.y < col_size) && (threadIdx.x == 0))
    {
        mask_dev[col_idx * gridDim.y + blockIdx.x] = block_mask[threadIdx.y]; // 4 bytes
        // combined copy to global memory
        // mask_dev[(blockIdx.y * gridDim.y + blockIdx.x) * TILE_DIM + threadIdx.y] = block_mask[threadIdx.y]; 
    }
}

void nms(int *keep, const float *boxes, const int n, const int m,
         const int max_out, const float thresh, int *num_out, int device_id)
{
    _set_device(device_id);

    const int b_per_grid = DIVUP(n, TILE_DIM); // # of blocks per grid on one axis
    const dim3 grid_size(b_per_grid, b_per_grid);
    const dim3 block_size(TILE_DIM, TILE_DIM);

    float *boxes_dev;
    DTYPE *mask_dev;
    CHECK(cudaMalloc(&boxes_dev, n * m * sizeof(float)));
    CHECK(cudaMalloc(&mask_dev, n * b_per_grid * sizeof(DTYPE)));
    CHECK(cudaMemcpy(boxes_dev, boxes, n * m * sizeof(float), cudaMemcpyHostToDevice));

    nms_kernel<<<grid_size, block_size>>>(boxes_dev, mask_dev, n, thresh);

    std::vector<DTYPE> mask_host(n * b_per_grid);
    CHECK(cudaMemcpy(&mask_host[0],
                     mask_dev,
                     n * b_per_grid * sizeof(DTYPE),
                     cudaMemcpyDeviceToHost));

    // keep track of the removed bboxes
    std::vector<DTYPE> remv(b_per_grid, 0);
    
    // unwrap the mask_host and compare
    int num_keep_temp = 0;
    for (int i = 0; i < n; i++)
    {
        int nblock = i / TILE_DIM;
        int inblock = i % TILE_DIM;
        if (!(remv[nblock] & (1UL << inblock))) 
        {
            if (num_keep_temp < max_out)
            {
                keep[num_keep_temp++] = i;
                DTYPE *p = &mask_host[0] + i * b_per_grid;
                for (int j = nblock; j < b_per_grid; j++)
                {
                    remv[j] |= p[j];
                }
            } else {break;}
        }
    }
    *num_out = num_keep_temp;

    CHECK(cudaFree(boxes_dev));
    CHECK(cudaFree(mask_dev));
}
