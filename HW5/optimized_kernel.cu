#include <cuda_runtime.h>
#include <stdio.h>
#include "optimized_kernel.h"
#include "utilities.h"
#include "constants.h"

__global__ void sgemm_optimized_batched(int M_, int N_, int K_,
                                        const float *A_all,
                                        const float *B_all,
                                        float *C_blocks,
                                        int numA, int numB) {
    int pair = blockIdx.x; 
    int lane = threadIdx.x;

    int totalPairs = numA * numB;
    if (pair >= totalPairs || lane >= 16) return;

    int i = pair / numB;
    int j = pair % numB;

    const float* A = A_all + i * 16;
    const float* B = B_all + j * 16;

    // Shared Memory (The "Workbench")
    __shared__ float s_A[16];
    __shared__ float s_B[16];

    // Load from Global to Shared (Same as before)
    // Using float4 for speed
    float4* s_A_vec = reinterpret_cast<float4*>(s_A);
    float4* s_B_vec = reinterpret_cast<float4*>(s_B);
    const float4* A_vec = reinterpret_cast<const float4*>(A);
    const float4* B_vec = reinterpret_cast<const float4*>(B);

    if (lane < 4) s_A_vec[lane] = A_vec[lane]; 
    else if (lane < 8) s_B_vec[lane - 4] = B_vec[lane - 4];

    __syncthreads();

    int row = lane / 4;
    int col = lane % 4;

    float sum = 0.f;
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        sum += s_A[row * 4 + k] * s_B[k * 4 + col];
    }

    C_blocks[pair * 16 + lane] = sum;
}

void batched_pairwise_optimized(const float *d_As, const float *d_Bs,
                                float *d_C_sum,
                                int numA, int numB,
                                int M_, int N_, int K_) {
    int numPairs = numA * numB;
    int totalC   = M_ * N_;

    float* d_C_pairs = nullptr;
    CHECK_CUDA(cudaMalloc(&d_C_pairs, (size_t)numPairs * totalC * sizeof(float)));

    dim3 grid(numPairs);
    dim3 block(16);

    sgemm_optimized_batched<<<grid, block>>>(M_, N_, K_, d_As, d_Bs, d_C_pairs, numA, numB);
    CHECK_CUDA(cudaDeviceSynchronize());

    int threads = 256;
    int blocks  = (totalC + threads - 1) / threads;
    reduce_blocks_sum<<<blocks, threads>>>(d_C_pairs, d_C_sum,
                                          numPairs, M_, N_);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaFree(d_C_pairs));
}