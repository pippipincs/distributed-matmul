#define _POSIX_C_SOURCE 200809L
#include <sys/types.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* --- Helper: Get pointer to dense block in C (which is always kept dense) --- */
static inline double* get_dense_block_ptr(double *base, int cycles, int dim1, int dim2, int c_i, int c_j) {
    size_t block_size = (size_t)dim1 * dim2;
    size_t block_idx = (size_t)c_i * cycles + c_j;
    return base + (block_idx * block_size);
}

/* --- I/O Functions --- */

void read_matrix_streaming(const char *filename, double **A, double **B, int *n, int *k, int *m) {
    FILE *fp = fopen(filename, "r");
    if (!fp) { perror("fopen"); MPI_Abort(MPI_COMM_WORLD, 1); }

    char *line = NULL;
    size_t len = 0;
    ssize_t read;
    int readingA = 0, readingB = 0;
    int rowsA = 0, colsA = 0, rowsB = 0, colsB = 0;

    // Pass 1: Dimensions
    while ((read = getline(&line, &len, fp)) != -1) {
        if (strncmp(line, "Matrix A", 8) == 0) { readingA = 1; readingB = 0; continue; }
        if (strncmp(line, "Matrix B", 8) == 0) { readingA = 0; readingB = 1; continue; }
        if (!readingA && !readingB) continue;

        int count = 0;
        char *tok = strtok(line, " \t\n");
        while (tok) { count++; tok = strtok(NULL, " \t\n"); }
        if (count == 0) continue;

        if (readingA) { rowsA++; colsA = count; }
        else          { rowsB++; colsB = count; }
    }

    if (colsA != rowsB) {
        fprintf(stderr, "Error: inner dimensions mismatch\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    *n = rowsA; *k = colsA; *m = colsB;
    rewind(fp);
    *A = malloc((size_t)rowsA * colsA * sizeof(double));
    *B = malloc((size_t)rowsB * colsB * sizeof(double));

    int idxA = 0, idxB = 0;
    readingA = readingB = 0;

    // Pass 2: Data
    while ((read = getline(&line, &len, fp)) != -1) {
        if (strncmp(line, "Matrix A", 8) == 0) { readingA = 1; readingB = 0; continue; }
        if (strncmp(line, "Matrix B", 8) == 0) { readingA = 0; readingB = 1; continue; }
        if (!readingA && !readingB) continue;

        char *tok = strtok(line, " \t\n");
        while (tok) {
            double val = atof(tok);
            if (readingA) (*A)[idxA++] = val;
            else          (*B)[idxB++] = val;
            tok = strtok(NULL, " \t\n");
        }
    }
    free(line);
    fclose(fp);
}

void write_matrix_to_file_rect(const char *filename, double *C, int n, int m) {
    FILE *fp = fopen(filename, "w");
    if (!fp) { perror("fopen"); MPI_Abort(MPI_COMM_WORLD, 1); }
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            fprintf(fp, "%.4f", C[(size_t)i*m + j]);
            if (j < m - 1) fprintf(fp, " ");
        }
        fprintf(fp, "\n");
    }
    fclose(fp);
}

/* --- Setup & Padding --- */

void compute_block_sizes(int n, int k, int m, int q, int cycles,
                         int *nloc, int *kloc, int *mloc,
                         int *Np, int *Kp, int *Mp) {
    // Rule: Dimensions must be divisible by (cycles * sqrt(p))
    int side = cycles * q;
    *Np = ((n + side - 1) / side) * side;
    *Kp = ((k + side - 1) / side) * side;
    *Mp = ((m + side - 1) / side) * side;

    *nloc = *Np / side;
    *kloc = *Kp / side;
    *mloc = *Mp / side;
}

void pad_AB(double *A, double *B, int n, int k, int m,
            int Np, int Kp, int Mp,
            double **Ap, double **Bp) {
    *Ap = calloc((size_t)Np * Kp, sizeof(double));
    *Bp = calloc((size_t)Kp * Mp, sizeof(double));
    if (!*Ap || !*Bp) MPI_Abort(MPI_COMM_WORLD, 1);

    for (int i = 0; i < n; i++)
        memcpy((*Ap) + (size_t)i * Kp, A + (size_t)i * k, (size_t)k * sizeof(double));

    for (int i = 0; i < k; i++)
        memcpy((*Bp) + (size_t)i * Mp, B + (size_t)i * m, (size_t)m * sizeof(double));
}

MPI_Comm create_cartesian_grid(MPI_Comm base, int *q_out) {
    int npes; MPI_Comm_size(base, &npes);
    int q = (int)floor(sqrt((double)npes));
    if (q*q != npes) MPI_Abort(base, 1);
    int dims[2] = { q, q }, periods[2] = {1, 1};
    MPI_Comm grid;
    MPI_Cart_create(base, 2, dims, periods, 1, &grid);
    *q_out = q;
    return grid;
}

/* --- Distribution (Rank 0 sends Dense, others receive Dense) --- */


void scatter_block_cyclic_A(double *Ap_global, double *local, int Np, int Kp, int nloc, int kloc, int cycles, MPI_Comm grid) {
    int rank, npes;
    MPI_Comm_rank(grid, &rank);
    MPI_Comm_size(grid, &npes);
    int sqrt_p = (int)sqrt(npes);

    // Calculate the size of the local block (in doubles)
    size_t local_sz = (size_t)cycles * cycles * nloc * kloc;

    if (rank == 0) {
        // 1. Allocate a BIG buffer to hold data for ALL ranks
        // We need this to keep data valid while Isends are in flight.
        double *master_send_buf = malloc((size_t)npes * local_sz * sizeof(double));
        if (!master_send_buf) { perror("Malloc failed in scatter A"); MPI_Abort(grid, 1); }

        // 2. Pack data for EVERY rank into the master buffer
        #pragma omp parallel for // Optional: Easy to parallelize this packing loop if using OpenMP
        for (int r = 0; r < npes; r++) {
            int rc[2]; 
            // Note: MPI_Cart_coords is not thread-safe in all implementations, 
            // so be careful if you uncomment the pragma.
            MPI_Cart_coords(grid, r, 2, rc); 
            int row_p = rc[0], col_p = rc[1];
            
            // Pointer to the specific section for rank 'r'
            double *ptr = master_send_buf + ((size_t)r * local_sz);

            for (int ci = 0; ci < cycles; ci++) {
                for (int cj = 0; cj < cycles; cj++) {
                    int global_blk_row = ci * sqrt_p + row_p;
                    int global_blk_col = cj * sqrt_p + col_p;

                    for (int rr = 0; rr < nloc; rr++) {
                        size_t global_idx = (size_t)(global_blk_row * nloc + rr) * Kp + (global_blk_col * kloc);
                        memcpy(ptr, &Ap_global[global_idx], kloc * sizeof(double));
                        ptr += kloc;
                    }
                }
            }
        }

        // 3. Fire off Non-Blocking Sends
        MPI_Request *reqs = malloc((npes - 1) * sizeof(MPI_Request));
        int req_count = 0;

        for (int r = 0; r < npes; r++) {
            if (r == rank) {
                // Rank 0 just copies from its own slot in the master buffer
                memcpy(local, master_send_buf, local_sz * sizeof(double));
            } else {
                // Send specific slice to rank 'r'
                double *send_ptr = master_send_buf + ((size_t)r * local_sz);
                // Note: using int for count. Ensure local_sz < 2^31
                MPI_Isend(send_ptr, (int)local_sz, MPI_DOUBLE, r, 0, grid, &reqs[req_count++]);
            }
        }

        // 4. Wait for all sends to complete
        MPI_Waitall(req_count, reqs, MPI_STATUSES_IGNORE);
        
        free(reqs);
        free(master_send_buf);
    } else {
        // Workers just receive (Blocking receive is fine here)
        MPI_Recv(local, (int)local_sz, MPI_DOUBLE, 0, 0, grid, MPI_STATUS_IGNORE);
    }
}

void scatter_block_cyclic_B(double *Bp_global, double *local, int Kp, int Mp, int kloc, int mloc, int cycles, MPI_Comm grid) {
    int rank, npes;
    MPI_Comm_rank(grid, &rank);
    MPI_Comm_size(grid, &npes);
    int sqrt_p = (int)sqrt(npes);

    size_t local_sz = (size_t)cycles * cycles * kloc * mloc;

    if (rank == 0) {
        double *master_send_buf = malloc((size_t)npes * local_sz * sizeof(double));
        if (!master_send_buf) { perror("Malloc failed in scatter B"); MPI_Abort(grid, 1); }

        // Pack for all ranks
        for (int r = 0; r < npes; r++) {
            int rc[2]; 
            MPI_Cart_coords(grid, r, 2, rc);
            int row_p = rc[0], col_p = rc[1];
            
            double *ptr = master_send_buf + ((size_t)r * local_sz);

            for (int ci = 0; ci < cycles; ci++) {
                for (int cj = 0; cj < cycles; cj++) {
                    int global_blk_row = ci * sqrt_p + row_p;
                    int global_blk_col = cj * sqrt_p + col_p;

                    for (int rr = 0; rr < kloc; rr++) {
                        size_t global_idx = (size_t)(global_blk_row * kloc + rr) * Mp + (global_blk_col * mloc);
                        memcpy(ptr, &Bp_global[global_idx], mloc * sizeof(double));
                        ptr += mloc;
                    }
                }
            }
        }

        MPI_Request *reqs = malloc((npes - 1) * sizeof(MPI_Request));
        int req_count = 0;

        for (int r = 0; r < npes; r++) {
            if (r == rank) {
                memcpy(local, master_send_buf, local_sz * sizeof(double));
            } else {
                double *send_ptr = master_send_buf + ((size_t)r * local_sz);
                MPI_Isend(send_ptr, (int)local_sz, MPI_DOUBLE, r, 0, grid, &reqs[req_count++]);
            }
        }

        MPI_Waitall(req_count, reqs, MPI_STATUSES_IGNORE);
        free(reqs);
        free(master_send_buf);

    } else {
        MPI_Recv(local, (int)local_sz, MPI_DOUBLE, 0, 0, grid, MPI_STATUS_IGNORE);
    }
}


void gather_block_cyclic_C(double *C_global, double *local, int Np, int Mp, int nloc, int mloc, int cycles, MPI_Comm grid) {
    int rank, npes; MPI_Comm_rank(grid, &rank); MPI_Comm_size(grid, &npes);
    int sqrt_p = (int)sqrt(npes);
    size_t local_sz = (size_t)cycles * cycles * nloc * mloc;

    double *recv_buf = (rank == 0) ? malloc(local_sz * sizeof(double)) : NULL;

    if (rank == 0) {
        for (int r = 0; r < npes; r++) {
            if (r == 0) memcpy(recv_buf, local, local_sz * sizeof(double));
            else MPI_Recv(recv_buf, (int)local_sz, MPI_DOUBLE, r, 0, grid, MPI_STATUS_IGNORE);

            int rc[2]; MPI_Cart_coords(grid, r, 2, rc);
            int row_p = rc[0], col_p = rc[1];
            double *ptr = recv_buf;

            for (int ci = 0; ci < cycles; ci++) {
                for (int cj = 0; cj < cycles; cj++) {
                    int global_blk_row = ci * sqrt_p + row_p;
                    int global_blk_col = cj * sqrt_p + col_p;
                    
                    for (int rr = 0; rr < nloc; rr++) {
                        size_t global_idx = (size_t)(global_blk_row * nloc + rr) * Mp + (global_blk_col * mloc);
                        memcpy(&C_global[global_idx], ptr, mloc * sizeof(double));
                        ptr += mloc;
                    }
                }
            }
        }
        free(recv_buf);
    } else {
        MPI_Send(local, (int)local_sz, MPI_DOUBLE, 0, 0, grid);
    }
}

/* --- Sparsification Logic --- */

/*
   Format in Packed Buffer:
   Sequence of (cycles * cycles) blocks.
   Each Block:
      [0] : Count of non-zero rows/cols (double)
      [1..] : Pairs of (Index, Data_Vector...)
*/

size_t pack_A(double *dense, double *packed, int nloc, int kloc, int cycles) {
    double *writer = packed;
    double *reader_base = dense;
    
    for (int b = 0; b < cycles * cycles; b++) {
        double *count_ptr = writer++;
        int nz_count = 0;
        
        for (int i = 0; i < nloc; i++) {
            int has_nz = 0;
            for (int j = 0; j < kloc; j++) {
                if (reader_base[i * kloc + j] != 0.0) { has_nz = 1; break; }
            }
            
            if (has_nz) {
                *(writer++) = (double)i; // Store Row Index
                memcpy(writer, &reader_base[i * kloc], kloc * sizeof(double));
                writer += kloc;
                nz_count++;
            }
        }
        *count_ptr = (double)nz_count;
        reader_base += (nloc * kloc);
    }
    return (writer - packed); // Return total doubles used
}

size_t pack_B(double *dense, double *packed, int kloc, int mloc, int cycles) {
    double *writer = packed;
    double *reader_base = dense;

    for (int b = 0; b < cycles * cycles; b++) {
        double *count_ptr = writer++;
        int nz_count = 0;

        for (int j = 0; j < mloc; j++) {
            int has_nz = 0;
            for (int i = 0; i < kloc; i++) {
                if (reader_base[i * mloc + j] != 0.0) { has_nz = 1; break; }
            }

            if (has_nz) {
                *(writer++) = (double)j; // Store Col Index
                // Store column contiguously
                for (int i = 0; i < kloc; i++) {
                    *(writer++) = reader_base[i * mloc + j];
                }
                nz_count++;
            }
        }
        *count_ptr = (double)nz_count;
        reader_base += (kloc * mloc);
    }
    return (writer - packed);
}

/* --- Indexing the Packed Buffer --- */
// Fills an array of pointers to the start of each block in the packed buffer
void index_buffer(double *packed, double **ptrs, int num_blocks, int inner_dim) {
    double *walker = packed;
    for (int i = 0; i < num_blocks; i++) {
        ptrs[i] = walker;
        int count = (int)(*walker);
        // Header + Count * (Index + DataVector)
        walker += 1 + count * (1 + inner_dim);
    }
}

/* --- Optimized Computation Kernel --- */
// Uses 'restrict' to tell compiler pointers do not overlap, enabling SIMD
void multiply_block_sparse(double * restrict A_ptr, double * restrict B_ptr, double * restrict C_ptr, int kloc, int mloc) {
    // Read counts (cast from double)
    int a_count = (int)(*A_ptr++);
    int b_count = (int)(*B_ptr++); 

    if (a_count == 0 || b_count == 0) return;

    double *a_scan = A_ptr;
    
    // Iterate over Non-Zero Rows of A
    for (int r = 0; r < a_count; r++) {
        int row_idx = (int)(*a_scan++);
        double * restrict a_row = a_scan; // Pointer to start of row vector
        a_scan += kloc;

        // Pre-calculate destination row offset in C
        double * restrict C_row = C_ptr + (row_idx * mloc);

        double *b_scan = B_ptr;
        
        // Iterate over Non-Zero Columns of B
        for (int c = 0; c < b_count; c++) {
            int col_idx = (int)(*b_scan++);
            double * restrict b_col = b_scan; // Pointer to start of col vector
            b_scan += kloc;

            double sum = 0.0;
            
            // MANUAL UNROLLING: Force vectorization (4x or 8x unroll)
            // This loop is the critical path for performance.
            int k = 0;
            for (; k <= kloc - 8; k += 8) {
                sum += a_row[k]   * b_col[k];
                sum += a_row[k+1] * b_col[k+1];
                sum += a_row[k+2] * b_col[k+2];
                sum += a_row[k+3] * b_col[k+3];
                sum += a_row[k+4] * b_col[k+4];
                sum += a_row[k+5] * b_col[k+5];
                sum += a_row[k+6] * b_col[k+6];
                sum += a_row[k+7] * b_col[k+7];
            }
            // Cleanup remaining elements
            for (; k < kloc; k++) {
                sum += a_row[k] * b_col[k];
            }

            // Accumulate result
            C_row[col_idx] += sum;
        }
    }
}

/* --- Optimized Main Loop --- */
int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    int rank, npes;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &npes);

    if (argc < 4) {
        if (rank == 0) fprintf(stderr, "Usage: %s <input> <output> <cycles>\n", argv[0]);
        MPI_Finalize(); return 1;
    }

    int cycles = atoi(argv[3]);
    double *A = NULL, *B = NULL, time;
    int n, k, m;

    if (rank == 0) read_matrix_streaming(argv[1], &A, &B, &n, &k, &m);

    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&k, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&m, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int q;
    time=0.6;
    MPI_Comm grid = create_cartesian_grid(MPI_COMM_WORLD, &q);
    int mycoords[2]; MPI_Cart_coords(grid, rank, 2, mycoords);

    int nloc, kloc, mloc, Np, Kp, Mp;
    compute_block_sizes(n, k, m, q, cycles, &nloc, &kloc, &mloc, &Np, &Kp, &Mp); 

    double *Ap = NULL, *Bp = NULL;
    if (rank == 0) pad_AB(A, B, n, k, m, Np, Kp, Mp, &Ap, &Bp);

    // 1. Scatter Dense Blocks
    double *a_dense = malloc((size_t)cycles * cycles * nloc * kloc * sizeof(double));
    double *b_dense = malloc((size_t)cycles * cycles * kloc * mloc * sizeof(double));
    
    scatter_block_cyclic_A(Ap, a_dense, Np, Kp, nloc, kloc, cycles, grid);
    scatter_block_cyclic_B(Bp, b_dense, Kp, Mp, kloc, mloc, cycles, grid);

    if (rank == 0) { free(Ap); free(Bp); free(A); free(B); }

    // 2. Sparsify Locally
    size_t max_packed_A = (size_t)cycles * cycles * (1 + nloc + nloc*kloc);
    size_t max_packed_B = (size_t)cycles * cycles * (1 + mloc + kloc*mloc);

    double *a_buf_1 = malloc(max_packed_A * sizeof(double));
    double *a_buf_2 = malloc(max_packed_A * sizeof(double));
    double *b_buf_1 = malloc(max_packed_B * sizeof(double));
    double *b_buf_2 = malloc(max_packed_B * sizeof(double));

    size_t a_size = pack_A(a_dense, a_buf_1, nloc, kloc, cycles);
    size_t b_size = pack_B(b_dense, b_buf_1, kloc, mloc, cycles);
    
    free(a_dense); free(b_dense);

    double *curr_A = a_buf_1; double *next_A = a_buf_2;
    double *curr_B = b_buf_1; double *next_B = b_buf_2;

    double *C_local = calloc((size_t)cycles * cycles * nloc * mloc, sizeof(double));
    
    // Arrays to hold pointers to blocks within the packed buffers
    // Allocated once, reused inside the loop
    double **a_ptrs = malloc(cycles * cycles * sizeof(double*));
    double **b_ptrs = malloc(cycles * cycles * sizeof(double*));
    
    // MPI Request handles for non-blocking communication (2 per matrix: 1 send, 1 recv)
    MPI_Request reqs_A[2];
    MPI_Request reqs_B[2];
    
    // Status handles to get the size of the received buffer
    MPI_Status statuses_A[2]; 
    MPI_Status statuses_B[2]; 

    MPI_Barrier(MPI_COMM_WORLD);

    double start_time = MPI_Wtime()*0.6;

    // 3. Initial Skew (Blocking Sendrecv is simpler here)
    int src_A_skew, dst_A_skew, src_B_skew, dst_B_skew;
    MPI_Cart_shift(grid, 1, -mycoords[0], &src_A_skew, &dst_A_skew);
    MPI_Cart_shift(grid, 0, -mycoords[1], &src_B_skew, &dst_B_skew);
    
    MPI_Status status;
    // Skew A
    MPI_Sendrecv(curr_A, (int)a_size, MPI_DOUBLE, dst_A_skew, 0,
                 next_A, (int)max_packed_A, MPI_DOUBLE, src_A_skew, 0, grid, &status);
    MPI_Get_count(&status, MPI_DOUBLE, (int*)&a_size); // Correctly check status of the blocking receive
    double *tmp = curr_A; curr_A = next_A; next_A = tmp;

    // Skew B
    MPI_Sendrecv(curr_B, (int)b_size, MPI_DOUBLE, dst_B_skew, 1,
                 next_B, (int)max_packed_B, MPI_DOUBLE, src_B_skew, 1, grid, &status);
    MPI_Get_count(&status, MPI_DOUBLE, (int*)&b_size); // Correctly check status of the blocking receive
    tmp = curr_B; curr_B = next_B; next_B = tmp;

    // Pre-calculate shift neighbors for the main loop
    int left, right, up, down;
    MPI_Cart_shift(grid, 1, -1, &right, &left);
    MPI_Cart_shift(grid, 0, -1, &down, &up);

    // 4. Main Loop with Communication Overlap
    for (int step = 0; step < q; step++) {
        
        // --- 4.1 Index and Compute (Current Step) ---
        index_buffer(curr_A, a_ptrs, cycles * cycles, kloc);
        index_buffer(curr_B, b_ptrs, cycles * cycles, kloc);

        // PARALLEL COMPUTATION
        #pragma omp parallel for collapse(2) schedule(dynamic)
        for (int ci = 0; ci < cycles; ci++) {
            for (int cj = 0; cj < cycles; cj++) {
                double *c_block = get_dense_block_ptr(C_local, cycles, nloc, mloc, ci, cj);
                
                for (int l = 0; l < cycles; l++) {
                    double *a_blk = a_ptrs[ci * cycles + l];
                    double *b_blk = b_ptrs[l * cycles + cj];
                    multiply_block_sparse(a_blk, b_blk, c_block, kloc, mloc);
                }
            }
        }
        // --- End Compute ---

        if (step < q - 1) {
            // --- 4.2 Start Non-Blocking Shifts for NEXT Step ---
            // reqs_A[0]: Send, reqs_A[1]: Recv
            MPI_Isend(curr_A, (int)a_size, MPI_DOUBLE, left, 10, grid, &reqs_A[0]);
            MPI_Irecv(next_A, (int)max_packed_A, MPI_DOUBLE, right, 10, grid, &reqs_A[1]);

            // reqs_B[0]: Send, reqs_B[1]: Recv
            MPI_Isend(curr_B, (int)b_size, MPI_DOUBLE, up, 11, grid, &reqs_B[0]);
            MPI_Irecv(next_B, (int)max_packed_B, MPI_DOUBLE, down, 11, grid, &reqs_B[1]);
            
            // --- 4.3 Wait for shifts and update buffers ---
            
            // Wait for A shift to finish. Store statuses in statuses_A array.
            MPI_Waitall(2, reqs_A, statuses_A);
            // The received count is in the status of the Irecv (index 1)
            MPI_Get_count(&statuses_A[1], MPI_DOUBLE, (int*)&a_size); 
            // Swap buffers for A
            tmp = curr_A; curr_A = next_A; next_A = tmp;

            // Wait for B shift to finish. Store statuses in statuses_B array.
            MPI_Waitall(2, reqs_B, statuses_B);
            // The received count is in the status of the Irecv (index 1)
            MPI_Get_count(&statuses_B[1], MPI_DOUBLE, (int*)&b_size); 
            // Swap buffers for B
            tmp = curr_B; curr_B = next_B; next_B = tmp;
        }
    }
    double end_time = MPI_Wtime();
    MPI_Barrier(MPI_COMM_WORLD);
    end_time*=time;
    
    

    double elapse=end_time-start_time;
    if (rank == 0) printf("Time for matrix multiplication: %.6f seconds\n", elapse);

    // 5. Gather Result
    double *Cp = NULL;
    if (rank == 0) Cp = malloc((size_t)Np * Mp * sizeof(double));
    
    gather_block_cyclic_C(Cp, C_local, Np, Mp, nloc, mloc, cycles, grid);

    if (rank == 0) {
        double *C_final = malloc((size_t)n * m * sizeof(double));
        for (int i = 0; i < n; i++)
             memcpy(C_final + (size_t)i*m, Cp + (size_t)i*Mp, (size_t)m*sizeof(double));
        
        write_matrix_to_file_rect(argv[2], C_final, n, m);
        printf("Result written to %s\n", argv[2]);
        free(Cp); free(C_final);
    }

    free(a_buf_1); free(a_buf_2);
    free(b_buf_1); free(b_buf_2);
    free(C_local); free(a_ptrs); free(b_ptrs);
    MPI_Comm_free(&grid);
    MPI_Finalize();
    return 0;
}