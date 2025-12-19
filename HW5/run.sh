nvcc -O3 -arch=sm_75 \
    main.cu utilities.cu naive_kernel.cu optimized_kernel.cu \
    -o batched_gemm && ./batched_gemm && rm -f batched_gemm