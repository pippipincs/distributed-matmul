#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

#include "naive_kernel.h"
#include "optimized_kernel.h"
#include "utilities.h"
#include "constants.h"


int main() {
  const int m = M, n = N, k = K;
  const int numA = NUM_A;
  const int numB = NUM_B;
  const int numPairs = numA * numB;
  const int matA_elems = m * k;
  const int matB_elems = k * n;
  const int matC_elems = m * n;

  printf("Batched pairwise GEMM test\n");
  printf("M=%d, N=%d, K=%d, numA=%d, numB=%d, numPairs=%d\n",
         m, n, k, numA, numB, numPairs);

  // Host allocations
  float *h_As = (float *)malloc((size_t)numA * matA_elems * sizeof(float));
  float *h_Bs = (float *)malloc((size_t)numB * matB_elems * sizeof(float));
  float *h_C_naive = (float *)malloc(matC_elems * sizeof(float));
  float *h_C_opt = (float *)malloc(matC_elems * sizeof(float));

  if (!h_As || !h_Bs || !h_C_naive || !h_C_opt) {
    fprintf(stderr, "Host malloc failed\n");
    return EXIT_FAILURE;
  }

  init_host_matrices(h_As, h_Bs, numA, numB, m, n, k);

  // Device allocations
  float *d_As = NULL, *d_Bs = NULL, *d_C_naive = NULL, *d_C_opt = NULL;
  CHECK_CUDA(cudaMalloc((void **)&d_As,
                        (size_t)numA * matA_elems * sizeof(float)));
  CHECK_CUDA(cudaMalloc((void **)&d_Bs,
                        (size_t)numB * matB_elems * sizeof(float)));
  CHECK_CUDA(cudaMalloc((void **)&d_C_naive,
                        matC_elems * sizeof(float)));
  CHECK_CUDA(cudaMalloc((void **)&d_C_opt,
                        matC_elems * sizeof(float)));

  CHECK_CUDA(cudaMemcpy(d_As, h_As,
                        (size_t)numA * matA_elems * sizeof(float),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_Bs, h_Bs,
                        (size_t)numB * matB_elems * sizeof(float),
                        cudaMemcpyHostToDevice));

  // CUDA events for timing
  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));

  float time_naive_ms = 0.0f;
  float time_opt_ms = 0.0f;


  float best_speedup = 0.0f;

  for (int trial = 0; trial < 5; ++trial) {
    CHECK_CUDA(cudaEventRecord(start));
    batched_pairwise_naive(d_As, d_Bs, d_C_naive,
                            numA, numB, m, n, k);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&time_naive_ms, start, stop));

    CHECK_CUDA(cudaEventRecord(start));
    batched_pairwise_optimized(d_As, d_Bs, d_C_opt,
                                numA, numB, m, n, k);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&time_opt_ms, start, stop));

    float speedup = time_naive_ms / time_opt_ms;
    if (speedup > best_speedup) {
      best_speedup = speedup;
    }
  }

  // Copy results back and compare
  CHECK_CUDA(cudaMemcpy(h_C_naive, d_C_naive,
                        matC_elems * sizeof(float),
                        cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(h_C_opt, d_C_opt,
                        matC_elems * sizeof(float),
                        cudaMemcpyDeviceToHost));

  double max_abs_diff = 0.0;
  double max_rel_diff = 0.0;

  for (int idx = 0; idx < matC_elems; ++idx) {
    double ref  = (double)h_C_naive[idx];
    double diff = fabs(ref - (double)h_C_opt[idx]);

    if (diff > max_abs_diff) {
      max_abs_diff = diff;
    }

    double denom = fabs(ref);
    double rel = (denom > 1e-6) ? diff / denom : diff;
    if (rel > max_rel_diff) {
      max_rel_diff = rel;
    }
  }

  // correctness score
  int correctness_score = 0;
  const double abs_tol = 1e-3;
  const double rel_tol = 1e-4;

  if (max_abs_diff < abs_tol || max_rel_diff < rel_tol) {
    correctness_score = 5;
  } else {
    correctness_score = 0;
  }

  printf("  Max rel error         : %.6e\n", max_rel_diff);


  // performance: speedup of (2) compared to (1)
  float two_speedup = best_speedup;
  float perf_score = 0.0f;

  if (two_speedup > 1.0f) {
    perf_score = 10.0f * two_speedup / HARD_CODED_SPEEDUP;
    if (perf_score > 10.0f) perf_score = 10.0f;
  } else {
    perf_score = 0.0f;
  }

  // if correctness failed, zero performance score
  if (correctness_score == 0) {
    perf_score = 0.0f;
  }

  float total_score = (float)correctness_score + perf_score;

  printf("\nResults:\n");
  printf("  Speedup (naive/opt) : %.3f x\n", two_speedup);
  printf("\nGrading:\n");
  printf("  Correctness score (0 or 5): %d\n", correctness_score);
  printf("  Performance score (0..10)  : %.2f (HARD_CODED_SPEEDUP=%.2f)\n",
         perf_score, HARD_CODED_SPEEDUP);
  printf("  Total score (0..15)       : %.2f\n", total_score);

  // cleanup
  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));
  CHECK_CUDA(cudaFree(d_As));
  CHECK_CUDA(cudaFree(d_Bs));
  CHECK_CUDA(cudaFree(d_C_naive));
  CHECK_CUDA(cudaFree(d_C_opt));
  free(h_As);
  free(h_Bs);
  free(h_C_naive);
  free(h_C_opt);

  return 0;
}
