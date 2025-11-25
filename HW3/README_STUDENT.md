# HW3: Sparse Block-Cyclic Cannon’s Algorithm

## Overview

This directory contains the materials for completing HW3, where you will extend the previous implementation of Cannon's algorithm to a block-cyclic sparse version of Cannon's. The provided files and tests will help you verify correctness and performance as you generalize your implementation.



## 1. Provided Materials

You are given the following files:
- `README_STUDENT.md` – This file. Contains the necessary information to complete this assignment.
- `variable_cannon.c` – A working implementation to run cannon's algorithm in non-cyclic fashion without any sparsity.
- `tests/` – Directory containing input/output files for all test cases (there are no hidden tests for this assignment). Test `i` may be found at `tests/test_{i}`. A single test directory contains an input file `in.txt` and output file `out.txt`. The input file contains both matrices `A` and `B`. The output file contains the expected matrix `C`. 
- `autograder.sh` – Bash script to compile and test your code across all tests we will consider. This contains the exact set of commands we will run for each of the `tests`.
- `autograde_runner.py` – Script to compute your final score. This reads in the log outputs of `autograder.sh` to determine how many speedup/unit tests your program passes.


## 2. Your Task

Your job is to update the version of MPI Cannon's algorithm present inside of `variable_cannon.c` in a new file which implements the same algorithm - but introduces sparsity & adds block-cyclic output distribution. See below for a high level view of what must be programmed. Your eventual result should pass all tests present inside of `autograder.sh`.

1. You must change the current implementation of cannon's algorithm to be block-cyclic. The current implementation decomposes the output matrix (C = AB) using a 2-d decomposition of sqrt(p) x sqrt(p) blocks of C, where `p` is the number of processes. The updated implementation should instead decompose C into (k x sqrt(p)) x (k x sqrt(p)) blocks of C, where `k` is the number of cycles (you can assume that `p` is a square number). A given process will then be responsible for `k^2` blocks of C. As an example, if k=2, then each process should have one block in the upper left quadrant, one in upper right, one in the lower right quadrant, and one in the lower left quadrant.

2. In order to ensure that the distributions of elements in A and B are properly sized within this block-cyclic configuration, be sure to update the padding such that the number of rows/cols in both matrices is divisible by (k x sqrt(p)). The padding should be distributed among blocks so that the last ranks in each row/column do not get stuck with a large number of 0s.

3. You must not sparsify the blocks before distributing from the process with rank 0. In other words, the general logic of distribution of blocks of A,B to the other processes should be largely unchanged from the current implementation such that even 0's are still sent - you will just have to account for the block-cyclic nature of the distribution.

4. Both A and B should be loaded row-wise.

5. You should not use routing to separate implementations for special cases - you should have one implementation of this algorithm which is general enough across potential inputs.

6. For sparsity, A will have some rows which are all 0. B will have some columns which are all 0.

7. Each process, after receiving their k^2 blocks of A and k^2 blocks of B from rank 0, should immediately update the representations of these blocks so that they only contain non-zero information. To do this, you will only need to store the nonzero data, plus the list of whichever rows (A) or columns (B) are nonzero. As an example, if we use a 4 process cannon's with 1 cycle on Test 0, then process 0 would have the upper left corner of A. In this case, this corresponds to [[0, 0], [-7, -3]]. To store this, you would only need to store [[-7, -3]] and [1], where [1] corresponds the row which has non-zero entries. In total, a given process will have to store `k` lists of 0 rows in A, and `k` lists of 0 cols in B.

8. In order to actually perform Cannon's algorithm (corresponding to `MatrixMatrixMultiply_rect` in the current implementation), you should only communicate the sparse representations of the data. Note that you do not need to communicate which rows are sparse (A) or which columns (B) as each process will get this information when sparsifying the matrices. In other words, each step of Cannon's will only require that you send non-zero data - the block dimensions and nonzero rows/cols are fixed on each process.

9. To perform the computation, you must skip any row/column computation which has zero rows in A or zero columns in B. If you have performed the sparsification well, then the computation should look quite similar to the triply nested matrix computation in the original file.

10. For each of the k^2 blocks that a given process must compute of C, the output dimensions will be known after sparsifying the distributed data. You can use this to allocate memory only one time.

11. In the `MatrixMatrixMultiply_rect` function of the original implementation of this algorithm, there are a total of `2 * sqrt(p)` communication steps required to perform computation (sqrt(p) for A, sqrt(p) for B) - these may be found inside of the for loop of this function. This number of communications should be maintained in your implementation. In other words, you should not have an inner for loop to separately communicate each of the `k` blocks of A and `k` blocks of B - these should all be communicated at one time.

12. Your implementation should then see speedups by skipping over computations involving 0s (sparsity), and ensuring a better load balancing (block-cyclic).

Once complete, you should pass all the unit and speedup tests inside of `autograder.sh`. 

As a tip, try working with 4x4, 4x8, 8x4, 8x8 matrices using 2 cycles, sparsity, and 4 processes by hand (Tests 0 and 1 are good starting points for this). Write out which processes should have which elements, then walk through what should happen on each iteration of Cannons (there will only be two iterations after the initial skew step). After you have a good grasp of this, work through the program itself.

## 3. Getting Started
- **Ensure that you have downloaded the zipfile for this assignment from the course website**
- **Create your implementation:**  
  You will edit `sparse_cyclic_cannon.c` for your solution.

- **Create a hostfile:**  
  Make a file named `hosts.txt` listing the node you will use (i.e., your assigned plate node). The number of slots you should use in this file is 64.


## 4. Running a Single Test

To run a specific test manually (in this case, 4 processes on Test 0 with 2 cycles):
```bash
mpicc -O3 -std=c99 -o build/sparse_cyclic_cannon sparse_cyclic_cannon.c -lm
mpirun -np 4 --hostfile hosts.txt --map-by node build/sparse_cyclic_cannon/test_0/in.txt build/out_test0_np4.txt 2
```

If successful, this will output a print statement to the shell like:
```
Time for matrix multiplication: %.6f seconds\n
```
where `%.6f` is replaced by a floating point number. Additionally, this will create a file named `build/out_test0_np4.txt`. The contents of this should match that found in `tests/test_0/out.txt`.

This example uses Test 0. Replace the input file (`tests/test_0/in.txt`) and output file (`build/out_test0_np4.txt`) as needed to check the performance of other tests.


## 5. Running the Full Autograder Script

To run all tests and see pass/fail results:
```bash
./autograder.sh
```


## 6. Running the Full Autograde Runner (for a Score)

To run the autograder and receive a scored grade in addition to the logs:
```bash
python3 autograde_runner.py
```


## 7. What’s the Difference?

- **autograder.sh:**  
  Compiles your code and runs all tests, showing which tests pass or fail.
- **autograde_runner.py:**  
  Runs the autograder script and computes your overall score from `autograder.sh`, as will be reported for grading by the autograder. This is what we will use for getting your canvas grades.


## 8. Test Descriptions

The tests are grouped as follows (see `autograder.sh` for details):

- **Small matrix tests (0-3)**  
  Small matrices with varying dimensions & levels of sparsity. These are not run for timings, but instead for correctness
- **Speedup Test (4)**  
  A single input test is tested against various number of processes.

Each test uses input files in `tests/test_X/in.txt` and compares your output to `tests/test_X/out.txt`.


## 9. Autograder Schedule

The autograder will be run approximately every 24 hours, starting from **Monday, November 17**. Make sure your code is ready and passes all tests before the deadline. To submit go to the [Canvas Link](https://canvas.umn.edu/courses/518528/assignments/4943253).


## 10.Grading

Your final score will be a composition of an autograded portion & a manually graded portion. The autograded portion will be worth 12 points:

- **Autograder Total Score:** 12 points.
- **Autograder Deductions:**
  - For each test that results in a timeout or runtime error:  
    **−(8 / 37) points** per failure (where 37 is the total number of tests).
  - For each unit test that fails (incorrect output):  
    **−(7 / 37) points** per failure.
  - For each speedup test that fails (test 4):  
    **−(4 / 5) points** per failure (where 5 is the number of speedup checks).
- **Autograder Minimum Score:** Your score will not go below 0.
- **Notes:**
   - Compilation failure results in a score of 0.
   - The autograder will run all tests and apply penalties as described above.
   - Your autograder score on canvas is reported out of 12.
   - The autograder will run every ~24 hours.

The manual portion will be worth 3 points. This portion will result from your report pdf and should contain only the following (i.e. the report does not need to go into the details of your implementation, just respond to the below points):

- **Manual Total Score:** 3 points.
- **Manual Grading Rubric:** 
  - **(.75 points)** Only 2*sqrt(p) communications are used in the for loop of your implementation of Cannon's Block-cyclic Sparse.
  - **(.75 points)** Provide the empirical Speedup & Efficiency metrics when running your program on Test 4 with 2 cycles at np={4,9,16,25}. To calculate the serial time, use the original `variable_cannon.c` implementation with 1 process as a benchmark.
  - **(.75 points)** As you vary the number of cycles at {1, 2, 3, 4}, what happens to your speedups on Test 4 (for this, you can fix np to be 4)? Give a reason around why the results you see occur.
  - **(.75 points)** Why do we need to use both block-cyclic and sparse representations? Why is neither alone sufficient to achieve the necessary speedups we are after?


## 11. Submission

This assignment requires the submission of two files to [Canvas](https://canvas.umn.edu/courses/518528/assignments/4943253) by **Nov 28, 11:59PM**:

- `sparse_cyclic_cannon.c` --> Your implementation of a sparse, cyclic version of Cannon's algorithm.
- `report.pdf` --> A report containing your answers to the questions of the manual grading section above.