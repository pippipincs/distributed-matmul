#include <cuda_runtime.h>
#include "utilities.h"

__global__ void reduce_blocks_sum(const float *C_blocks, float *C_sum,
                                  int numBlocks, int M_, int N_) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = M_ * N_;
  if (idx >= total) return;

  float sum = 0.0f;
  for (int b = 0; b < numBlocks; ++b) {
    sum += C_blocks[(size_t)b * total + idx];
  }
  C_sum[idx] = sum;
}

static float init_val(int mat_idx, int row, int col) {
  return 0.001f * (float)(mat_idx + 1) +
         0.01f * (float)row +
         0.1f  * (float)col;
}

void init_host_matrices(float *h_As, float *h_Bs,
                        int numA, int numB,
                        int M_, int N_, int K_) {
  for (int a = 0; a < numA; ++a) {
    for (int r = 0; r < M_; ++r) {
      for (int c = 0; c < K_; ++c) {
        h_As[(size_t)a * M_ * K_ + r * K_ + c] =
            init_val(a, r, c);
      }
    }
  }

  for (int b = 0; b < numB; ++b) {
    for (int r = 0; r < K_; ++r) {
      for (int c = 0; c < N_; ++c) {
        h_Bs[(size_t)b * K_ * N_ + r * N_ + c] =
            init_val(b + 1000, r, c);
      }
    }
  }
}