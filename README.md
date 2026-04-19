# Distributed Matrix Multiplication with Cannon's Algorithm

A progressively optimized implementation of Cannon's algorithm for distributed matrix multiplication using MPI. Starting from a basic square-matrix version, the project extends to arbitrary dimensions, block-cyclic distribution, and sparse computation — scaling to 64 processes on 1000x1000+ matrices.

## Project Structure

```
.
├── arbitrary-cannon/          # Dense Cannon's for arbitrary matrix dimensions
│   ├── square_cannon.c        # Starter code (square matrices only)
│   ├── variable_cannon.c      # Extended to non-square & non-divisible dimensions
│   ├── autograder.sh
│   └── tests/
└── blockcyclic-cannon/        # Sparse block-cyclic Cannon's
    ├── variable_cannon.c      # Dense baseline for timing comparison
    ├── sparse_cyclic_cannon.c # Full sparse block-cyclic implementation
    ├── autograder.sh
    └── tests/
```

## Part 1: Arbitrary-Dimension Cannon's (`arbitrary-cannon/`)

Extends the classic Cannon's algorithm to support matrices of any dimension, not just square matrices with sides divisible by sqrt(p).

**Key techniques:**
- **Zero-padding** — matrices are padded so dimensions become divisible by sqrt(p), then trimmed on output
- **2D Cartesian grid** — `MPI_Cart_create` with periodic boundaries for wraparound shifts
- **Subarray datatypes** — `MPI_Type_create_subarray` to scatter/gather rectangular blocks from the global matrix
- **Sendrecv rotation** — initial skew alignment followed by sqrt(p) shift-and-multiply steps

**Supports:** 1, 4, 9, 16, 25, 36, 49, or 64 MPI processes with matrices of any size.

### Build & Run

```bash
cd arbitrary-cannon
mpicc -O3 -std=c99 -o build/variable_cannon variable_cannon.c -lm
mpirun -np 4 --hostfile hosts.txt --map-by node \
    build/variable_cannon tests/test_0/in.txt build/out.txt
```

## Part 2: Sparse Block-Cyclic Cannon's (`blockcyclic-cannon/`)

Builds on Part 1 with two major optimizations: block-cyclic distribution for better load balancing and a custom sparse representation to skip zero computations.

**Key techniques:**
- **Block-cyclic distribution** — each process owns k^2 blocks arranged in a tiled pattern across the matrix, where k is the number of cycles. Padding is extended so dimensions are divisible by (k * sqrt(p))
- **Custom packed sparse format** — after distribution, each process sparsifies its local blocks:
  - Matrix A: stores only non-zero rows as `[count | (row_index, row_data) | ...]`
  - Matrix B: stores only non-zero columns as `[count | (col_index, col_data) | ...]`
- **Sparse-only communication** — shift operations send only the packed (non-zero) data, with `MPI_Get_count` to determine received sizes
- **Communication overlap** — non-blocking `MPI_Isend`/`MPI_Irecv` for shifts overlapped with computation via double buffering
- **Optimized multiply kernel** — `restrict` pointers and 8x manual loop unrolling for vectorization; only iterates over non-zero row/column pairs

**Communication cost:** 2 * sqrt(p) point-to-point operations per Cannon's loop (unchanged from the dense version — all k blocks are communicated in a single message per shift).

### Build & Run

```bash
cd blockcyclic-cannon
mpicc -O3 -std=c99 -o build/sparse_cyclic_cannon sparse_cyclic_cannon.c -lm
mpirun -np 4 --hostfile hosts.txt --map-by node \
    build/sparse_cyclic_cannon tests/test_0/in.txt build/out.txt 2
```

The third argument (`2`) is the number of cycles (k).

## Testing

Each part includes an autograder:

```bash
./autograder.sh            # run all tests, show pass/fail
python3 autograde_runner.py  # run tests and compute score
```

## Why Both Block-Cyclic and Sparse?

- **Block-cyclic alone** distributes work more evenly across processes but still computes over zeros — no reduction in total FLOPs.
- **Sparse alone** skips zero computation but with a standard 2D decomposition, some processes may hold mostly-zero blocks while others hold dense blocks — creating load imbalance.
- **Combined**, block-cyclic ensures each process gets a representative mix of sparse and dense regions, while the sparse format eliminates unnecessary computation and communication within each block.

## Technologies

C, MPI (`MPI_Cart`, `MPI_Isend`/`MPI_Irecv`, `MPI_Type_create_subarray`), OpenMP (optional parallelism over cycle blocks)
