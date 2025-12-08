# **CSCI 5451 — Homework 4: Profiling GPU Convolution Kernels**  
**Due Date:** **December 7, 2025**  
**Canvas Submission Link:** https://canvas.umn.edu/courses/518528/assignments/4943254  
**Total Points:** 15

---

## **Overview**

In this assignment, you will analyze and profile a set of CUDA GPU kernels implementing a 2-D convolution. The provided code contains **four different convolution kernels**, each using a different optimization strategy. You will:

- Benchmark these kernels on **Google Colab GPUs** or on the provided cuda lab machines (csel-cuda-0x.cselabs.umn.edu)
- Explore how block sizes and coarsening factors change performance,  
- Interpret your results,  
- And compare performance across GPU architectures (T4 vs A100).

You **must understand the C code in detail**, so read it carefully before starting.

---

# **Using Google Colab for This Homework**

This assignment **must be completed in Google Colab** or on the **provided university lab machines** (csel-cuda-0x.cselabs.umn.edu).

### **To use the lab machines**
The following cuda machines (each containing a single T4 GPU) are now working and are available for use

 - csel-cuda-01.cselabs.umn.edu
 - csel-cuda-03.cselabs.umn.edu
 - csel-cuda-04.cselabs.umn.edu
 - csel-cuda-05.cselabs.umn.edu

Note that csel-cuda-02.cselabs.umn.edu is **not** available. If this is your assigned cuda machine, then you will have to use one of the other machines. Please select at random from the above list. In order to run the tests for this assignment, you will need to download the zip file, unzip, navigate to the corresponding directory, then run the following:

```
nvcc -Xptxas -O3 -O3 -arch=sm_75 convolution.cu -o convolution_hw
./convolution_hw
```

### **To use Colab:**

1. Go to **https://colab.research.google.com/**
2. Create a new notebook
3. Go to:
   **Runtime → Change runtime type**
4. Set:
   - **Hardware Accelerator: GPU**
   - For this assignment, we will make use of both A100 and T4 GPUs
5. Upload the following file into the notebook using the provided starter cells:
   - `convolution.cu` (the provided source file)

Your `.ipynb` **must include the following three cells only**, with no modification except switching between T4 and A100 lines as instructed:

---

### **Cell 1: Upload the CUDA Program**
```python
from google.colab import files
uploaded = files.upload()
```

---

### **Cell 2: Compile**
For **T4 (sm_75)**:
```python
!nvcc -Xptxas -O3 -O3 -arch=sm_75 convolution.cu -o convolution_hw
```

For **A100 (sm_80)**:
```python
!nvcc -Xptxas -O3 -O3 -arch=sm_80 convolution.cu -o convolution_hw
```

---

### **Cell 3: Run**
```python
!./convolution_hw
```

---

# **Assignment Tasks**

## **(1) Block-size parameter sweep (3 points)**

For each of the four kernels:

- Vary `dim3 block(x, y)` across a reasonable grid of values.  
  Examples (you may choose your own):  
  ```
  x ∈ {8, 16, 32}  
  y ∈ {8, 16, 32}
  ```
- Record the runtime for each `(x, y)` pair.
- Produce a **2-D table per kernel** showing speeds for each pairings for each kernel.

---

## **(2) Vary the thread-coarsening factor for `function_d` (3 points)**

`function_d` uses thread coarsening such that more than one output is computed per thread:

You must test the following coarsening pairs `(OPT_COARSEN_Y, OPT_COARSEN_X)`:

```
(1,1), (2,1), (4,1), (8,1),
(1,2), (2,2), (4,2), (8,2),
(1,4), (2,4), (4,4), (8,4),
(1,8), (2,8), (4,8), (8,8)
```

For each pair:

- Recompile with the new macros using `dim3 block(16, 16).
- If you experience errors, you should give a detailed accounting as to why the error occurs.
- Produce a table of speeds for each of these configurations of coarsening factors when fixing the threadblock size to be 

---

## **(3) Explain the performance ordering (5 points)**

Determine the order of kernels in terms of which is fastest

Provide a **thorough explanation** of why the kernels perform in the order that they do. This should be grounded in the hardware itself as well as constraints imposed by the software.

---

## **(4) Fastest kernel comparison on T4 vs A100 (2 points)**

The following is a table of timings (in ms) for a T4 and A100 GPU with a fixed configuration of the macro definitions (`OPT_BLOCK_W`, `OPT_COARSEN_X`, etc.). In other words, they are running the exact same problem, only the hardware has changed.

| Function    | GPU A     | GPU B   |
|-------------|-----------|---------|
| function_a  | 12.253    | 2.333   |
| function_b  | 11.73     | 2.787   |
| function_c  | 23.50     | 6.957   |
| function_d  | 6.78      | 1.242   |

You must first determine which column corresponds to the timings on a T4 GPU, and which on an A100 GPU. Then give a **hardware-based explanation** of why the the GPU you chose is faster grounded in concepts we have covered in the course lectures.

---

## **(5) Short Questions (2 points, 0.5 each)**

### **(a)**  
**What is the theoretical peak arithmetic intensity of this program?**  
Compute FLOPs per byte of matrix data loaded (you may ignore the convolution filter's memory footprint).

---

### **(b)**  
**Why does the benchmarking code use a warmup phase?**  
Look up “GPU warmup” or “kernel warmup” in benchmarking practice.

---

### **(c)**  
**What does the `-arch` flag control when compiling the CUDA program?**  
Also:  
**Why do we use `-O3` optimization flags?**

---

### **(d)**  
**What are some additional ways you could further speed up this program?**  

---

# **Submission Instructions**

You must submit **one file** to Canvas:

### **(1) A PDF containing:**

- All 2-D tables (from Parts 1 & 2)
- Explanations (Part 3)
- GPU comparison results (Part 4)
- Answers to the short questions (Part 5)

**Canvas link:**  
https://canvas.umn.edu/courses/518528/assignments/4943254

---

# **Provided Files**

You will receive a ZIP archive containing:

- `HW4.md` (this file)  
- `convolution.cu` (the GPU program you must analyze)  
- `profile_convolution.ipynb` (with the three required Colab cells - detailed above)

---
