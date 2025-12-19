#include <cuda_runtime.h>
#include <stdio.h>
#include "utilities.h" 

// ------------------------------------------------------------------
// 1. HELPER: Vectorized Float4 Math (Raw Implementation)
// ------------------------------------------------------------------
__device__ inline float4 add_float4(float4 a, float4 b) {
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

// ------------------------------------------------------------------
// 2. KERNEL: Reduce (Sum) Matrices 
// Optimized with Block-Level Reduction & Vectorized Loads
// ------------------------------------------------------------------
__global__ void reduce_matrices_kernel(const float4* __restrict__ input,
                                       float4* output,
                                       int num_matrices) {
    // Each block is responsible for summing ONE row (float4) of the matrices
    // Grid Dim: 4 (since M=4, we need 4 rows)
    int row_idx = blockIdx.x; 

    // Initialize accumulator
    float4 local_sum = make_float4(0.f, 0.f, 0.f, 0.f);

    // Grid-Stride Loop: Iterate through all matrices for this specific row
    // We stride by blockDim.x so threads coalesce reads as much as possible
    for (int i = threadIdx.x; i < num_matrices; i += blockDim.x) {
        // Input is flat: [Matrix0_Row0, Matrix0_Row1... | Matrix1_Row0...]
        // Stride is 4 float4s per matrix
        local_sum = add_float4(local_sum, input[i * 4 + row_idx]);
    }

    // Shared Memory "Workbench" for Reduction
    __shared__ float4 s_data[256]; // Assumes blockDim.x <= 256
    int tid = threadIdx.x;
    s_data[tid] = local_sum;
    __syncthreads();

    // Parallel Tree Reduction in Shared Memory
    // Unrolling the last few iterations manually is a common optimization,
    // but a loop is cleaner and sufficiently fast here.
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_data[tid] = add_float4(s_data[tid], s_data[tid + s]);
        }
        __syncthreads();
    }

    // Write the result for this row to global memory
    if (tid == 0) {
        output[row_idx] = s_data[0];
    }
}

// ------------------------------------------------------------------
// 3. KERNEL: Final 4x4 Multiplication
// Replaces the CPU step. Runs on 1 Block, 16 Threads.
// ------------------------------------------------------------------
__global__ void final_mult_kernel(const float4* __restrict__ sum_A, 
                                  const float4* __restrict__ sum_B, 
                                  float* C_result) {
    // We only need 16 threads (one for each element of the 4x4 result)
    int tid = threadIdx.x;
    if (tid >= 16) return;

    int row = tid / 4;
    int col = tid % 4;

    // Load Sum_A and Sum_B into Shared Memory for fast broadcast
    // (Optional for such small data, but good practice)
    __shared__ float4 s_A[4];
    __shared__ float4 s_B[4];

    // Cooperative Load
    if (tid < 4) {
        s_A[tid] = sum_A[tid];
        s_B[tid] = sum_B[tid];
    }
    __syncthreads();

    // Standard Matrix Multiply: C[row, col] = Dot(A[row], B_col)
    // We have A rows as float4. B is also stored as rows.
    
    float4 A_row_vec = s_A[row];
    
    // We need column 'col' from B. 
    // B is stored row-major in float4. B[0] is row 0.
    float b0 = (col==0)?s_B[0].x : (col==1)?s_B[0].y : (col==2)?s_B[0].z : s_B[0].w;
    float b1 = (col==0)?s_B[1].x : (col==1)?s_B[1].y : (col==2)?s_B[1].z : s_B[1].w;
    float b2 = (col==0)?s_B[2].x : (col==1)?s_B[2].y : (col==2)?s_B[2].z : s_B[2].w;
    float b3 = (col==0)?s_B[3].x : (col==1)?s_B[3].y : (col==2)?s_B[3].z : s_B[3].w;

    float res = A_row_vec.x * b0 + 
                A_row_vec.y * b1 + 
                A_row_vec.z * b2 + 
                A_row_vec.w * b3;

    C_result[tid] = res;
}

// ------------------------------------------------------------------
// HOST FUNCTION
// ------------------------------------------------------------------
void batched_pairwise_optimized(const float *d_As, const float *d_Bs,
                                float *d_C_sum,
                                int numA, int numB,
                                int M_, int N_, int K_) {
    // 1. Alloc Temps
    float4 *d_SumA, *d_SumB;
    CHECK_CUDA(cudaMalloc(&d_SumA, 4 * sizeof(float4)));
    CHECK_CUDA(cudaMalloc(&d_SumB, 4 * sizeof(float4)));

    // 2. Reduce A and B separately
    // Grid=4 (1 block per row), Block=256 threads
    reduce_matrices_kernel<<<4, 256>>>(reinterpret_cast<const float4*>(d_As), d_SumA, numA);
    reduce_matrices_kernel<<<4, 256>>>(reinterpret_cast<const float4*>(d_Bs), d_SumB, numB);

    // 3. Final Multiply on GPU
    final_mult_kernel<<<1, 16>>>(d_SumA, d_SumB, d_C_sum);
    
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaFree(d_SumA));
    CHECK_CUDA(cudaFree(d_SumB));
}