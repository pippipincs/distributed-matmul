# **HW5: Pair-Wise Batch GEMM Kernel**

**Due:** December 21  
**Submission Link:** https://canvas.umn.edu/courses/518528/assignments/4943255

---

## **Group Submission**

This assignment is a group submission with the same groups as used for your final project. Only one member of your group needs to submit this assignment. This score will apply to all students in your group.

---

## **Overview**

In this assignment, you will implement an optimized CUDA kernel for performing **pair-wise batched matrix multiplication**. Given:

- A list of matrices $A_0, A_1, \dots, A_{N_A-1}$
- A list of matrices $B_0, B_1, ..., B_{N_B-1}$

each of size **4×4**, your goal is to compute:

$
C = \sum_{i=0}^{N_A - 1} \sum_{j=0}^{N_B - 1} A_i B_j
$

This requires multiplying **every pair** $A_i, B_j$ and summing their results.

There are **no scaling coefficients**, just plain matrix multiplication with accumulation.

Your job is to dramatically reduce global memory traffic, increase arithmetic intensity, and build a high-performance CUDA kernel.

---

## **📁 File Structure & Editing Rules (Important)**

Your starter code is split across several `.cu` and `.h` files.

### ✔ You must only modify one file:

optimized_kernel.cu


This is also the **only file you will submit**.

It contains:

- The kernel `sgemm_optimized_batched_kernel`
- The wrapper `batched_pairwise_optimized(..)`

### ❌ Do **not** modify any other project files:

- `main.cu`
- `common.h`
- `utilities.[ch]`
- `naive_kernel.[ch]`

Changing these will likely break the autograder and result in a **score of 0**.

The provided version of `optimized_kernel.cu` initially duplicates the naive functionality so the project runs immediately. You must replace it with an optimized implementation. You can update this file any way you like, so long as the autograder functions properly and your function signature for `batched_pairwise_optimized` remains the same as in the present file.

In order to run the autograder locally, be sure that you are on your assigned csel-cuda lab machine, which will have a T4 GPU. You can run the autograder with 

```
chmod +u+x run.sh
./run.sh
```

---

## **🎯 Problem Definition: All-Pairs Matrix Multiplication**

You are given:

- $512$ matrices $A_i \in \mathbb{R}^{4 \times 4}$
- $512$ matrices $B_j \in \mathbb{R}^{4 \times 4}$

Your task is to compute:

$$
C = \sum_{i=0}^{511} \sum_{j=0}^{511} A_i B_j
$$

- Each multiplication produces a **4×4** matrix.
- You must sum all of them into a **single** 4×4 output matrix.
- There are **262,144** matrix multiplications.

Your kernel must compute this efficiently.

---

## **🔒 Hard-Coded Dimensions**

This assignment is **not** about writing a general GEMM implementation.

All matrices are **4×4**, and the batch sizes are fixed:

M = 4
N = 4
K = 4
NUM_A = 512
NUM_B = 512


Anything that increases speed is allowed *as long as* the results are correct and you **do not** make use of any more complex cuda library functions (cuBLAS, tensor cores) which abstract shared memory, coarsening, vectorized loading away from you.

---

## **🚀 Strategies for High Performance**

To maximize performance, focus on:

### **1. Increasing arithmetic intensity**
Load data once, reuse it many times.

### **2. Efficient block-level reductions**
Make use of reductions both within and between threadblocks for improved speedups.

### **3. Vectorized memory access**
Use `float4` to perform aligned, coalesced loads.

### **4. Shared Memory**
Be sure to make extensive use of shared memory constructs.

---

## **🧪 Autograding & Scoring**

The autograder evaluates:

### **✔ 5 points — Correctness**

- Your output must match the expected result within tolerance.
- If your code fails to compile or crashes:
  - **Correctness = 0**
  - **Performance = 0**
  - **Total Score = 0**

### **✔ 10 points — Performance**

The autograder runs the benchmark **5 times** and uses the **maximum speedup**:

$$
\text{speedup} = \frac{T_\text{naive}}{T_\text{optimized}}
$$


Let:

- `S` = your measured speedup  
- `S₀` = instructor-defined benchmark (we are using a speedup of 100x. If you achieve at least 100x speedups over the naive kernel, you will get full credit.) 

Your performance score is:

$$
\text{score}_\text{perf}
= \begin{cases}
0, & S \le 1 \\
10 \cdot \frac{S}{S_0}, & 1 < S < S_0 + 1 \\
10, & S \ge S_0 + 1
\end{cases}
$$


Total assignment score:

$$
\text{score} = \text{correctness (0–5)} + \text{performance (0–10)}
$$


The autograder prints all timing information and computed scores.

---

## **📦 Submission Instructions**

You must submit exactly one file:

optimized_kernel.cu


Do **not** compress it.  
Do **not** rename it.  
Do **not** modify other files.

The autograder compiles using:

```bash
nvcc -O3 -arch=sm_75 \
    main.cu utilities.cu naive_kernel.cu optimized_kernel.cu \
    -o batched_gemm && ./batched_gemm && rm -f batched_gemm
```

Your file must compile cleanly under this exact command.

## 📌 Final Notes

The assignment is 100% autograded.

Only modify optimized_kernel.cu.

Hardcoding optimizations is encouraged.

Shared memory, tiling, vectorized loads, and reduction patterns will be essential for speed.

Readability is appreciated, but performance matters most.