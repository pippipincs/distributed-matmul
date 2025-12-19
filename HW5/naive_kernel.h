#pragma once

void batched_pairwise_naive(const float *d_As, const float *d_Bs,
                            float *d_C_sum,
                            int numA, int numB,
                            int M, int N, int K);
