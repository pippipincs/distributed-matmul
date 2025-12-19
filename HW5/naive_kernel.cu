#include <cuda_runtime.h>
#include <stdio.h>
#include "naive_kernel.h"
#include "utilities.h"
#include "constants.h"

__global__ void naive_pairwise_kernel(const float* __restrict__ A_all,
                                      const float* __restrict__ B_all,
                                      float* __restrict__ C_pairs,
                                      int numA, int numB)
{
    int pair = blockIdx.x; 
    int lane = threadIdx.x;

    int totalPairs = numA * numB;
    if (pair >= totalPairs || lane >= 16) return;

    int i = pair / numB;
    int j = pair % numB;

    const float* A = A_all + i * 16;
    const float* B = B_all + j * 16;

    int row = lane / 4;
    int col = lane % 4;

    float sum = 0.f;
    for (int k = 0; k < 4; ++k)
        sum += A[row * 4 + k] * B[k * 4 + col];

    C_pairs[pair * 16 + lane] = sum;
}

void batched_pairwise_naive(const float *d_As, const float *d_Bs,
                            float *d_C_sum,
                            int numA, int numB,
                            int M_, int N_, int K_)
{
    int numPairs = numA * numB;
    int totalC   = M_ * N_;

    float* d_C_pairs = nullptr;
    CHECK_CUDA(cudaMalloc(&d_C_pairs, (size_t)numPairs * totalC * sizeof(float)));

    dim3 grid(numPairs);
    dim3 block(16);

    naive_pairwise_kernel<<<grid, block>>>(d_As, d_Bs, d_C_pairs, numA, numB);
    CHECK_CUDA(cudaDeviceSynchronize());

    int threads = 256;
    int blocks  = (totalC + threads - 1) / threads;
    reduce_blocks_sum<<<blocks, threads>>>(d_C_pairs, d_C_sum,
                                          numPairs, M_, N_);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaFree(d_C_pairs));
}
