
# HW2: Distributed Matrix Multiplication with Cannon’s Algorithm

## Overview

This directory contains the materials for completing HW2, where you will implement and extend Cannon’s algorithm for distributed matrix multiplication using MPI. The provided files and tests will help you verify correctness and performance as you generalize your implementation.


## 1. Provided Materials

You are given the following files:
- `README_STUDENT.md` – This file. Contains the necessary information to complete this assignment.
- `square_cannon.c` – Starter C implementation of Cannon's algorithm for square matrices. This allows you to compute `C=AB`, given some input text file containing both `A` and `B`. This starter code can successfully compute Tests 0-4, given that correct numbers of processes are chosen (see `autograder.sh` to see what numbers of processes should be used for each test). Read further for example usage.
- `tests/` – Directory containing input/output files for all test cases (there are no hidden tests for this assignment). Test `i` may be found at `tests/test_{i}`. A single test directory contains an input file `in.txt` and output file `out.txt`. The input file contains both matrices `A` and `B`. The output file contains the expected matrix `C`. 
- `autograder.sh` – Bash script to compile and test your code across all tests we will consider. This contains the exact set of commands we will run for each of the `tests`.
- `autograde_runner.py` – Script to compute your final score. This reads in the log outputs of `autograder.sh` to determine how many speedup/unit tests your program passes.

These materials allow you to run Cannon’s algorithm for **square matrices** where the number of rows/columns is divisible by the square root of the number of processes (`sqrt(p)`). The currently provided code does not work in the cases where the rows/columns are not divisible by `sqrt(p)`, or where the two matrices A & B are not square.


## 2. Your Task

You must update the program to support **general matrix dimensions**. In all the tests of your program, you can assume that the number of rows/columns in `A,B` is at least `sqrt(p)`, and that we will always launch a square number of processes up to 64 (i.e. 1, 4, 9, 16, 25, 36, 49, 64). To make your task easier, you should complete the below in order:

- **(a) Non-square matrices:** Ensure your code works for matrices where the number of rows/columns is divisible by `sqrt(p)`, even if the matrices are not square. These correspond to Tests 5-7 (again, see `autograder.sh` to see what numbers of processes should be spawned for each test case).
- **(b) Arbitrary dimensions:** Generalize your code to handle matrices of any size, even when dimensions are not divisible by `sqrt(p)`. Tests 8-11 correspond to this case, where we will be using Test 11 to perform speedup tests. Once you have completed this portion, you should be able to run Tests 8-11 with any number of square processes in (1, 4, 9, 16, 25, 36, 49, 64). 
  **Hint:** For (b) use padding with zero's (0) to extend matrices so they fit the required block sizes. This will ensure that you do not have to use variable sized sends/receives, but can instead standardize the sending and receiving operations to a constant local block size.


## 3. Getting Started
- **Ensure that you have downloaded the zipfile for this assignment from the course website**
- **Create your implementation:**  
  Copy the starter file to begin:
  ```bash
  cp square_cannon.c variable_cannon.c
  ```
  You will edit `variable_cannon.c` for your solution.

- **Create a hostfile:**  
  Make a file named `hosts.txt` listing the node you will use (e.g., your assigned plate node). The number of slots you should use in this file is 64.


## 4. Running a Single Test

To run a specific test manually (in this case, 4 processes on Test 0):
```bash
mpicc -O3 -std=c99 -o build/variable_cannon variable_cannon.c -lm
mpirun -np 4 --hostfile hosts.txt --map-by node build/variable_cannon tests/test_0/in.txt build/out_test0_np4.txt
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

- **Square Matrix Tests:**  
  Multiplying square matrices of various sizes (e.g., (2x2)x(2x2), (5x5)x(5x5), (4x4)x(4x4), (10x10)x(10x10), (36x36)x(36x36)) with different process counts. These are Tests 0-4.
- **Divisible Matrix Tests:**  
  Non-square matrices where dimensions are divisible by `sqrt(p)` (e.g., (2x4) x (4x6), (3x6) x (6x9), (5x10) x (10x25)) with different process counts. These are Tests 5-7.
- **Prime Dimension Tests:**  
  Matrices with prime dimensions, not divisible by `sqrt(p)` (e.g., (11x13) x (13x17), (13x17) x (17x19), (17x19) x (19x23)). These are tests 8-10.
- **Timing Test:**  
  Large matrix multiplication (1009x1007 x 1007x1019) to check for speedup and performance. This is Test 11.

Each test uses input files in `tests/test_X/in.txt` and compares your output to `tests/test_X/out.txt`. For Tests 0-7, we only use a number of processes such that each dimension of both `A` and `B` is divisible by `sqrt(p)`.


## 9. Autograder Schedule

The autograder will be run approximately every 24 hours, starting from **Monday, October 20**. Make sure your code is ready and passes all tests before the deadline. To submit go to the [Canvas Link](https://canvas.umn.edu/courses/518528/assignments/4943252).


## 10.Grading

Your final score will be a composition of an autograded portion & a manually graded portion. The autograded portion will be worth 12 points:

- **Autograder Total Score:** 12 points.
- **Autograder Deductions:**
  - For each test that results in a timeout or runtime error:  
    **−(9 / 53) points** per failure (where 53 is the total number of tests).
  - For each unit test that fails (incorrect output):  
    **−(8 / 53) points** per failure.
  - For each speedup test that fails (test 11, performance check):  
    **−(3 / 8) points** per failure (where 8 is the number of speedup checks).
- **Autograder Minimum Score:** Your score will not go below 0.
- **Notes:**
   - Compilation failure results in a score of 0.
   - The autograder will run all tests and apply penalties as described above.
   - Your autograder score on canvas is reported out of 12.

The manual portion will be worth 3 points:

- **Manual Total Score:** 3 points.
- **Manul Grading Rubric:**
  - You must catalog & describe the set of changes made to your program. You should include program stubs which demonstrate these changes. Your report should be compiled into a pdf.


## 11. Submission

This assignment requires the submission of two files to [Canvas](https://canvas.umn.edu/courses/518528/assignments/4943252) by **Nov 2, 11:59PM**:

- `variable_cannon.c` --> Your implementation of Cannon's algorithm which can use a square number of processes (1, 4, 9, 16, 25, 36, 49, 64) to perform matrix multiplication with arbitrary size input matrices.
- `report.pdf` --> A report describing your implementation & the changes you made to get your variable implementation to work properly.