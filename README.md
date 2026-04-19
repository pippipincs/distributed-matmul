# Distributed Matrix Multiplication with Cannon's Algorithm

A progressively optimized implementation of Cannon's algorithm for distributed matrix multiplication using MPI. Starting from a basic square-matrix version, the project extends to arbitrary dimensions, block-cyclic distribution, and sparse computation — scaling to 64 processes on 1000x1000+ matrices.



## Part 1: Arbitrary-Dimension Cannon's (`arbitrary-cannon/`)

Extends the classic Cannon's algorithm to support matrices of any dimension, not just square matrices with sides divisible by sqrt(p).

**Key techniques:**
- **Zero-padding** — matrices are padded so dimensions become divisible by sqrt(p), then trimmed on output
- **2D Cartesian grid** — `MPI_Cart_create` with periodic boundaries for wraparound shifts
- **Subarray datatypes** — `MPI_Type_create_subarray` to scatter/gather rectangular blocks from the global matrix
- **Sendrecv rotation** — initial skew alignment followed by sqrt(p) shift-and-multiply steps

**Supports:** 1, 4, 9, 16, 25, 36, 49, or 64 MPI processes with matrices of any size.



## Part 2: Sparse Block-Cyclic Cannon's (`blockcyclic-cannon/`)

Builds on Part 1 with two major optimizations: block-cyclic distribution for better load balancing and a custom sparse representation to skip zero computations.

**Key techniques:**
- **Block-cyclic distribution** — each process owns k^2 blocks arranged in a tiled pattern across the matrix, where k is the number of cycles. Padding is extended so dimensions are divisible by (k * sqrt(p))
- **Custom packed sparse format** — after distribution, each process sparsifies its local blocks:
  - Matrix A: stores only non-zero rows as `[count | (row_index, row_data) | ...]`
  - Matrix B: stores only non-zero columns as `[count | (col_index, col_data) | ...]`
- **Sparse-only communication** — shift operations send only the packed (non-zero) data, with `MPI_Get_count` to determine received sizes
- **Communication overlap** — non-blocking `MPI_Isend`/`MPI_Irecv` for shifts overlapped with computation via double buffering



## Technologies

C, MPI (`MPI_Cart`, `MPI_Isend`/`MPI_Irecv`, `MPI_Type_create_subarray`), OpenMP (optional parallelism over cycle blocks)
