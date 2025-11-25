#define _POSIX_C_SOURCE 200809L
#include <sys/types.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>


void read_matrix_streaming(const char *filename,
                           double **A, double **B,
                           int *n, int *k, int *m)
{
    FILE *fp = fopen(filename, "r");
    if (!fp) { perror("fopen"); MPI_Abort(MPI_COMM_WORLD, 1); }

    char *line = NULL;
    size_t len = 0;
    ssize_t read;
    int readingA = 0, readingB = 0;
    int rowsA = 0, colsA = 0, rowsB = 0, colsB = 0;

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
        fprintf(stderr, "Error: inner dimensions mismatch (A:%dx%d, B:%dx%d)\n",
                rowsA, colsA, rowsB, colsB);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    *n = rowsA; *k = colsA; *m = colsB;

    rewind(fp);
    *A = malloc((size_t)rowsA * colsA * sizeof(double));
    *B = malloc((size_t)rowsB * colsB * sizeof(double));

    int idxA = 0, idxB = 0;
    readingA = readingB = 0;

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


void write_matrix_to_file_rect(const char *filename, double *C, int n, int m)
{
    FILE *fp = fopen(filename, "w");
    if (!fp) { perror("fopen"); MPI_Abort(MPI_COMM_WORLD, 1); }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            fprintf(fp, "%.4f", C[i*m + j]);
            if (j < m - 1) fprintf(fp, " ");
        }
        fprintf(fp, "\n");
    }
    fclose(fp);
}


static inline int iceil(int a, int b) { return (a + b - 1) / b; }

void compute_block_sizes_uniform(int n, int k, int m, int q,
                                 int *nloc, int *kloc, int *mloc,
                                 int *Np, int *Kp, int *Mp)
{
    *Np = ((n + q - 1) / q) * q;
    *Kp = ((k + q - 1) / q) * q;
    *Mp = ((m + q - 1) / q) * q;

    *nloc = *Np / q;
    *kloc = *Kp / q;
    *mloc = *Mp / q;
}

void pad_AB_uniform(double *A, double *B, int n, int k, int m,
                    int Np, int Kp, int Mp,
                    double **Ap, double **Bp)
{
    *Ap = calloc((size_t)Np * Kp, sizeof(double));
    *Bp = calloc((size_t)Kp * Mp, sizeof(double));
    if (!*Ap || !*Bp) { perror("calloc"); MPI_Abort(MPI_COMM_WORLD, 1); }

    for (int i = 0; i < n; i++)
        memcpy((*Ap) + (size_t)i * Kp, A + (size_t)i * k, (size_t)k * sizeof(double));

    for (int i = 0; i < k; i++)
        memcpy((*Bp) + (size_t)i * Mp, B + (size_t)i * m, (size_t)m * sizeof(double));
}

MPI_Comm create_cartesian_grid(MPI_Comm base_comm, int *q_out)
{
    int npes; MPI_Comm_size(base_comm, &npes);
    int q = (int)floor(sqrt((double)npes));
    if (q*q != npes) {
        fprintf(stderr, "Error: npes=%d not a perfect square.\n", npes);
        MPI_Abort(base_comm, 1);
    }

    int dims[2] = { q, q }, periods[2] = {1, 1};
    MPI_Comm grid;
    MPI_Cart_create(base_comm, 2, dims, periods, 1, &grid);
    if (q_out) *q_out = q;
    return grid;
}


void scatter_2d_blocks_rect(double *Ap_global, double *local,
                            int Np, int Kp, int nloc, int kloc,
                            MPI_Comm grid)
{
    int rank; MPI_Comm_rank(grid, &rank);
    int dims[2], periods[2], coords[2];
    MPI_Cart_get(grid, 2, dims, periods, coords);

    if (rank == 0) {
        int npes; MPI_Comm_size(grid, &npes);
        for (int r = 0; r < npes; r++) {
            int rc[2]; MPI_Cart_coords(grid, r, 2, rc);
            int sizes[2] = { Np, Kp }, subs[2] = { nloc, kloc };
            int starts[2] = { rc[0]*nloc, rc[1]*kloc };

            MPI_Datatype sub; 
            MPI_Type_create_subarray(2, sizes, subs, starts, MPI_ORDER_C, MPI_DOUBLE, &sub);
            MPI_Type_commit(&sub);

            if (r == 0) {
                for (int i = 0; i < nloc; i++) {
                    const double *src = Ap_global + (size_t)(starts[0]+i)*Kp + starts[1];
                    double *dst = local + (size_t)i * kloc;
                    memcpy(dst, src, (size_t)kloc * sizeof(double));
                }
            } else {
                MPI_Send(Ap_global, 1, sub, r, 999, grid);
            }
            MPI_Type_free(&sub);
        }
    } else {
        MPI_Recv(local, (size_t)nloc * kloc, MPI_DOUBLE, 0, 999, grid, MPI_STATUS_IGNORE);
    }
}

void scatter_2d_blocks_rect_B(double *Bp_global, double *local,
                              int Kp, int Mp, int kloc, int mloc,
                              MPI_Comm grid)
{
    int rank; MPI_Comm_rank(grid, &rank);
    int dims[2], periods[2], coords[2];
    MPI_Cart_get(grid, 2, dims, periods, coords);

    if (rank == 0) {
        int npes; MPI_Comm_size(grid, &npes);
        for (int r = 0; r < npes; r++) {
            int rc[2]; MPI_Cart_coords(grid, r, 2, rc);
            int sizes[2] = { Kp, Mp }, subs[2] = { kloc, mloc };
            int starts[2] = { rc[0]*kloc, rc[1]*mloc };

            MPI_Datatype sub; 
            MPI_Type_create_subarray(2, sizes, subs, starts, MPI_ORDER_C, MPI_DOUBLE, &sub);
            MPI_Type_commit(&sub);

            if (r == 0) {
                for (int i = 0; i < kloc; i++) {
                    const double *src = Bp_global + (size_t)(starts[0]+i)*Mp + starts[1];
                    double *dst = local + (size_t)i * mloc;
                    memcpy(dst, src, (size_t)mloc * sizeof(double));
                }
            } else {
                MPI_Send(Bp_global, 1, sub, r, 1999, grid);
            }
            MPI_Type_free(&sub);
        }
    } else {
        MPI_Recv(local, (size_t)kloc * mloc, MPI_DOUBLE, 0, 1999, grid, MPI_STATUS_IGNORE);
    }
}

void gather_2d_blocks_rect(double *local, double *Cp_global,
                           int Np, int Mp, int nloc, int mloc,
                           MPI_Comm grid)
{
    int rank; MPI_Comm_rank(grid, &rank);
    int dims[2], periods[2], coords[2];
    MPI_Cart_get(grid, 2, dims, periods, coords);

    if (rank == 0) {
        int npes; MPI_Comm_size(grid, &npes);
        for (int r = 0; r < npes; r++) {
            int rc[2]; MPI_Cart_coords(grid, r, 2, rc);
            int sizes[2] = { Np, Mp }, subs[2] = { nloc, mloc };
            int starts[2] = { rc[0]*nloc, rc[1]*mloc };

            MPI_Datatype sub;
            MPI_Type_create_subarray(2, sizes, subs, starts, MPI_ORDER_C, MPI_DOUBLE, &sub);
            MPI_Type_commit(&sub);

            if (r == 0) {
                for (int i = 0; i < nloc; i++) {
                    double *dst = Cp_global + (size_t)(starts[0]+i)*Mp + starts[1];
                    const double *src = local + (size_t)i * mloc;
                    memcpy(dst, src, (size_t)mloc * sizeof(double));
                }
            } else {
                MPI_Recv(Cp_global, 1, sub, r, 2999, grid, MPI_STATUS_IGNORE);
            }
            MPI_Type_free(&sub);
        }
    } else {
        MPI_Send(local, (size_t)nloc * mloc, MPI_DOUBLE, 0, 2999, grid);
    }
}


void MatrixMultiply_rect(int nrows, int ninner, int ncols,
                         const double *A, const double *B, double *C)
{
    for (int i = 0; i < nrows; i++) {
        for (int k = 0; k < ninner; k++) {
            double aik = A[(size_t)i * ninner + k];
            const double *Bk = B + (size_t)k * ncols;
            double *Ci = C + (size_t)i * ncols;
            for (int j = 0; j < ncols; j++)
                Ci[j] += aik * Bk[j];
        }
    }
}

void MatrixMatrixMultiply_rect(int q,
                               int nloc, int kloc, int mloc,
                               double *a, double *b, double *c,
                               MPI_Comm grid)
{
    int myrank, mycoords[2], dims[2], periods[2];
    MPI_Comm_rank(grid, &myrank);
    MPI_Cart_get(grid, 2, dims, periods, mycoords);

    int left, right, up, down;
    MPI_Cart_shift(grid, 1, +1, &left,  &right);
    MPI_Cart_shift(grid, 0, +1, &up,    &down);

    int shiftsource, shiftdest;
    MPI_Status status;

    /* Initial skew */
    MPI_Cart_shift(grid, 1, -mycoords[0], &shiftsource, &shiftdest);
    MPI_Sendrecv_replace(a, (size_t)nloc*kloc, MPI_DOUBLE,
                         shiftdest, 100, shiftsource, 100, grid, &status);

    MPI_Cart_shift(grid, 0, -mycoords[1], &shiftsource, &shiftdest);
    MPI_Sendrecv_replace(b, (size_t)kloc*mloc, MPI_DOUBLE,
                         shiftdest, 200, shiftsource, 200, grid, &status);

    for (int step = 0; step < q; step++) {
        MatrixMultiply_rect(nloc, kloc, mloc, a, b, c);

        MPI_Sendrecv_replace(a, (size_t)nloc*kloc, MPI_DOUBLE,
                             left, 1000+step, right, 1000+step, grid, &status);

        MPI_Sendrecv_replace(b, (size_t)kloc*mloc, MPI_DOUBLE,
                             up, 2000+step, down, 2000+step, grid, &status);
    }
}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    int rank, npes;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &npes);

    if (argc < 3) {
        if (rank == 0)
            fprintf(stderr, "Usage: %s <input_file> <output_file>\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

    const char *input_file = argv[1];
    const char *output_file = argv[2];

    double *A = NULL, *B = NULL;
    int n=0, k=0, m=0;

    if (rank == 0)
        read_matrix_streaming(input_file, &A, &B, &n, &k, &m);

    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&k, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&m, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int q;
    MPI_Comm grid = create_cartesian_grid(MPI_COMM_WORLD, &q);

    int nloc, kloc, mloc, Np, Kp, Mp;
    compute_block_sizes_uniform(n, k, m, q, &nloc, &kloc, &mloc, &Np, &Kp, &Mp);

    double *Ap = NULL, *Bp = NULL;
    if (rank == 0)
        pad_AB_uniform(A, B, n, k, m, Np, Kp, Mp, &Ap, &Bp);

    double *a_local = malloc((size_t)nloc*kloc*sizeof(double));
    double *b_local = malloc((size_t)kloc*mloc*sizeof(double));
    double *c_local = calloc((size_t)nloc*mloc, sizeof(double));

    scatter_2d_blocks_rect(Ap, a_local, Np, Kp, nloc, kloc, grid);
    scatter_2d_blocks_rect_B(Bp, b_local, Kp, Mp, kloc, mloc, grid);

    double start, end, elapsed;
    start = MPI_Wtime();
    MatrixMatrixMultiply_rect(q, nloc, kloc, mloc, a_local, b_local, c_local, grid);
    end = MPI_Wtime();
    if (rank == 0) {
        elapsed = end - start;
        printf("Time for matrix multiplication: %.6f seconds\n", elapsed);
    }

    double *Cp = NULL;
    if (rank == 0) Cp = malloc((size_t)Np * Mp * sizeof(double));
    gather_2d_blocks_rect(c_local, Cp, Np, Mp, nloc, mloc, grid);

    if (rank == 0) {
        double *C_trim = malloc((size_t)n * m * sizeof(double));
        for (int i = 0; i < n; i++)
            memcpy(C_trim + (size_t)i*m, Cp + (size_t)i*Mp, (size_t)m*sizeof(double));

        write_matrix_to_file_rect(output_file, C_trim, n, m);
        printf("Result written to %s\n", output_file);

        free(A); free(B);
        free(Ap); free(Bp);
        free(Cp);
        free(C_trim);
    }

    MPI_Comm_free(&grid);
    free(a_local); free(b_local); free(c_local);
    MPI_Finalize();
    return 0;
}
