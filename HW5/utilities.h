#pragma once

__global__ void reduce_blocks_sum(const float *C_blocks, float *C_sum,
                                  int numBlocks, int M_, int N_);
void init_host_matrices(float *h_As, float *h_Bs,
                        int numA, int numB,
                        int M_, int N_, int K_);
static float init_val(int mat_idx, int row, int col);

#define CHECK_CUDA(call)                                                      \
  do {                                                                        \
    cudaError_t _e = (call);                                                  \
    if (_e != cudaSuccess) {                                                  \
      printf("CUDA Error %s:%d: %s\n", __FILE__, __LINE__,                    \
             cudaGetErrorString(_e));                                         \
    }                                                                         \
  } while (0)

  #define CEIL_DIV(M_, N_) (((M_) + (N_) - 1) / (N_))