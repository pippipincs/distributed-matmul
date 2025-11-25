#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include <omp.h>
double min3(double a, double b, double c) {
   double m = a < b ? a : b;
   return m < c ? m : c;
}


void printMatrix(const char *name, double **matrix, int rows, int cols) {
   if (matrix == NULL || rows == 0 || cols == 0) {
       printf("%s is empty or not loaded.\n", name);
       return;
   }
   printf("%s (Rows: %d, Cols: %d):\n", name, rows, cols);
   for (int i = 0; i < rows; i++) {
       for (int j = 0; j < cols; j++) {
           printf("%.2f ", matrix[i][j]);
       }
       printf("\n");
   }
}




void freeMatrix(double ***matrix, int rows) {
   if (*matrix == NULL) return;
   for (int i = 0; i < rows; i++) {
       if ((*matrix)[i] != NULL) {
           free((*matrix)[i]);
       }
   }
   free(*matrix);
   *matrix = NULL;
}


int countIntegersInLine(char *line) {
   int cnt = 0;
   

   char *value=strtok(line," ");
   while (value!=NULL){
        value=strtok(NULL," ");
        cnt++;
   }
   return cnt;

   
   
}
int readMatricesFromFile(const char *filename,
                        double ***matrixA_ptr, int *rowsA_ptr, int *colsA_ptr,
                        double ***matrixB_ptr, int *rowsB_ptr, int *colsB_ptr) {


  
   *matrixA_ptr = NULL; *rowsA_ptr = 0; *colsA_ptr = 0;
   *matrixB_ptr = NULL; *rowsB_ptr = 0; *colsB_ptr = 0;


   FILE *file = fopen(filename, "r");
   if (file == NULL) {
       perror("Error opening file");
       return -1;
   }
   char *line = NULL;
    size_t len = 0;
   int readingMatrixA = 0;
   int readingMatrixB = 0;


   while (getline(&line, &len, file) != -1) {
       line[strcspn(line, "\n")] = 0;


       if (strcmp(line, "Matrix A") == 0) {
           readingMatrixA = 1;
           readingMatrixB = 0;
           *colsA_ptr = 0;
           continue;
       } else if (strcmp(line, "Matrix B") == 0) {
           readingMatrixB = 1;
           readingMatrixA = 0;
           *colsB_ptr = 0;
           continue;
       }


       if (readingMatrixA || readingMatrixB) {
            char *line_copy = strdup(line); 
        
            int current_num_cols = countIntegersInLine(line_copy);
            free(line_copy);
           
            

           if (current_num_cols > 0) {
              
               if (readingMatrixA && *colsA_ptr == 0) {
                   *colsA_ptr = current_num_cols;
               } else if (readingMatrixB && *colsB_ptr == 0) {
                   *colsB_ptr = current_num_cols;
               }


              
               if ((readingMatrixA && current_num_cols != *colsA_ptr) ||
                   (readingMatrixB && current_num_cols != *colsB_ptr)) {
                   fprintf(stderr, "Warning: Inconsistent column count for a matrix. Skipping line: %s\n", line);
                   continue;
               }


               double *currentRowData = (double *)malloc(current_num_cols * sizeof(double));
               if (currentRowData == NULL) {
                   perror("malloc error for row data");
                   fclose(file);
                  
                   freeMatrix(matrixA_ptr, *rowsA_ptr);
                   freeMatrix(matrixB_ptr, *rowsB_ptr);
                   return -2;
               }


               char *token_start = line;
               char *end_ptr;
               int col_idx = 0;
               long val;


               while (*token_start != '\0' && col_idx < current_num_cols) {
                   while (isspace((unsigned char)*token_start)) {
                       token_start++;
                   }
                   if (*token_start == '\0') break;


                   val = strtol(token_start, &end_ptr, 10);
                   if (end_ptr == token_start) {
                      
                       free(currentRowData);
                       break;
                   }


                   currentRowData[col_idx++] = (double)val;
                   token_start = end_ptr;
               }
              
              
               if (col_idx != current_num_cols) {
                   fprintf(stderr, "Warning: Malformed line or insufficient numbers for row. Skipping: %s\n", line);
                   free(currentRowData);
                   continue;
               }




               if (readingMatrixA) {
                   if (*rowsA_ptr == 0) {
                       *matrixA_ptr = (double **)malloc(sizeof(double *));
                       if (*matrixA_ptr == NULL) { perror("malloc error"); free(currentRowData); fclose(file); freeMatrix(matrixB_ptr, *rowsB_ptr); return -2; }
                   } else {
                       *matrixA_ptr = (double **)realloc(*matrixA_ptr, (*rowsA_ptr + 1) * sizeof(double *));
                       if (*matrixA_ptr == NULL) { perror("realloc error"); free(currentRowData); fclose(file); freeMatrix(matrixB_ptr, *rowsB_ptr); return -2; }
                   }
                   (*matrixA_ptr)[*rowsA_ptr] = currentRowData;
                   (*rowsA_ptr)++;
               } else if (readingMatrixB) {
                   if (*rowsB_ptr == 0) {
                       *matrixB_ptr = (double **)malloc(sizeof(double *));
                       if (*matrixB_ptr == NULL) { perror("malloc error"); free(currentRowData); fclose(file); freeMatrix(matrixA_ptr, *rowsA_ptr); return -2; }
                   } else {
                       *matrixB_ptr = (double **)realloc(*matrixB_ptr, (*rowsB_ptr + 1) * sizeof(double *));
                       if (*matrixB_ptr == NULL) { perror("realloc error"); free(currentRowData); fclose(file); freeMatrix(matrixA_ptr, *rowsA_ptr); return -2; }
                   }
                   (*matrixB_ptr)[*rowsB_ptr] = currentRowData;
                   (*rowsB_ptr)++;
               }
           } else {
              
               if ((readingMatrixA && *rowsA_ptr > 0) || (readingMatrixB && *rowsB_ptr > 0)) {
                   readingMatrixA = 0;
                   readingMatrixB = 0;
               }
           }
       }
       
   }
   

   fclose(file);
   return 0;
}


void compute_L2_matrix(double **A, double **B, double **D,
                      int mA, int mB, int n) {
  
   #pragma omp parallel for collapse(2)
   for (int i = 0; i < mA; i++) {
       for (int j = 0; j < mB; j++) {
           double sum = 0.0;
          
           for (int k = 0; k < n; k++) {
               double diff = A[i][k] - B[j][k];
               sum += diff * diff;
           }
           D[i][j] = sqrt(sum);
       }
   }
}
double* convertToSingleArray(double** matrix, int rows, int cols) {
    
    double* singleArray = (double*)malloc(rows * cols * sizeof(double));
    if (singleArray == NULL) {
        perror("Failed to allocate memory for single array");
        return NULL; 
    }

   
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            singleArray[i * cols + j] = matrix[i][j];
        }
    }

    return singleArray;
}


int main(int argc, char *argv[]){
   double **matrixA = NULL;
   int rowsA = 0, colsA = 0;
   double **matrixB = NULL;
   int rowsB = 0, colsB = 0;


   if (argc != 3) {
       printf("Usage: %s <matrix_file> <num_threads>\n", argv[0]);
       return 1;
   }


   char *test_name=argv[1];
   int num_threads=atoi(argv[2]);
  


   int result = readMatricesFromFile(test_name,
                                     &matrixA, &rowsA, &colsA,
                                     &matrixB, &rowsB, &colsB);

   double *matrixa=convertToSingleArray(matrixA,rowsA,colsA);
   double *matrixb=convertToSingleArray(matrixB,rowsB,colsB);
   if (result == -1) {
       fprintf(stderr, "Failed to open file: %s\n", test_name);
       return 1;
   } else if (result == -2) {
       fprintf(stderr, "Memory allocation failed during matrix reading.\n");
      
       return 1;
   }
   double start_time, end_time;
  
   double *dist = malloc(rowsA *rowsB* sizeof(double*));
   
   


   double *D = malloc(rowsA* rowsB * sizeof(double*));
   




   omp_set_num_threads(num_threads);
   start_time = omp_get_wtime();



   
   #pragma omp parallel for collapse(2) schedule(auto)
   for (int i = 0; i < rowsA; i++) {
       for (int j = 0; j < rowsB; j++) {
           double sum = 0.0;
          
           for (int k = 0; k < colsA; k++) {
               double diff = matrixa[i*colsA+k] - matrixb[j*colsA+k];
               sum += diff * diff;
           }
           dist[i*rowsB+j] = sqrt(sum);
   }
   }
  
   
    D[0] = dist[0];
        
    

   
  
   #pragma omp parallel
   {
   #pragma omp sections
   {
   #pragma omp section
   {
       for (int i = 1; i < rowsA; i++) {
           D[i*rowsB+0] = dist[i*rowsB+0] + D[(i - 1)*rowsB+0];
       }
   }


   #pragma omp section
   {
       for (int j = 1; j < rowsB; j++) {
           D[0*rowsB+j] = dist[0*rowsB+j] + D[0*rowsB+j - 1];
       }
   }
   }
   if (rowsA>=20){
    for (int k = 2; k < rowsA + rowsB - 1; k++) {
       #pragma omp for schedule(auto)
       for (int i = 1; i < rowsA; i++) {
           int j = k - i;


          
           if (j >= 1 && j < rowsB) {
               D[i*rowsB+j] = dist[i*rowsB+j] + min3(D[(i - 1)*rowsB+j], D[i*rowsB+j-1], D[(i - 1)*rowsB+j-1]);
           }
       }
       
    }
   }else{
    for (int k = 2; k < rowsA + rowsB - 1; k++) {
       
       for (int i = 1; i < rowsA; i++) {
           int j = k - i;


          
           if (j >= 1 && j < rowsB) {
               D[i*rowsB+j] = dist[i*rowsB+j] + min3(D[(i - 1)*rowsB+j], D[i*rowsB+j-1], D[(i - 1)*rowsB+j-1]);
           }
       }
       
    }
   }
    
   
    
   
   
   }


   end_time = omp_get_wtime();
   printf("\nThe Final Cost is: %f\n", D[(rowsA - 1)*rowsB+rowsB - 1]);
   double elapsed_time = end_time - start_time;


   printf("The Total Completion time (ms) is: %f \n", elapsed_time*1000);
   
   free(dist);
   free(D);
  
   freeMatrix(&matrixA, rowsA);
   freeMatrix(&matrixB, rowsB);


  




   return 0;
}





