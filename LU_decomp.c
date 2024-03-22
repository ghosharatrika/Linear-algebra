// This code calculates the LU decomposition of four matrices and hence demonstrate whether the decomposition is correct or not
#include <stdio.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_blas.h>

// Function to print a matrix row-wise
void print_matrix_row_wise(const gsl_matrix *m) {
    for (size_t i = 0; i < m->size1; ++i) {
        for (size_t j = 0; j < m->size2; ++j) {
            printf("%f\t", gsl_matrix_get(m, i, j));
        }
        printf("\n");
    }
}

// Function to perform LU decomposition of a matrix
void lu_decomposition(gsl_matrix *A) {
    gsl_permutation *p = gsl_permutation_alloc(A->size1); // Allocating permutation

    int s;
    gsl_linalg_LU_decomp(A, p, &s); // Performing LU decomposition

    gsl_matrix *L = gsl_matrix_alloc(A->size1, A->size1);
    gsl_matrix *U = gsl_matrix_alloc(A->size1, A->size1);

    // Extracting L and U matrices from the LU decomposition

    /*
       This for loop extracts the lower triangular matrix (L) and the upper triangular matrix (U) from the the modified A matrix. 
       Explicitly the diagonal elements of L are set to 1, as they are not stored during the LU decomposition in GSL.
    */
    for (size_t i = 0; i < A->size1; ++i) {
        for (size_t j = 0; j < A->size1; ++j) {
            if (i > j) {
                gsl_matrix_set(L, i, j, gsl_matrix_get(A, i, j));
                gsl_matrix_set(U, i, j, 0.0);
            } else {
                gsl_matrix_set(U, i, j, gsl_matrix_get(A, i, j));
                if (i == j) {
                    gsl_matrix_set(L, i, j, 1.0);   
                } else {
                    gsl_matrix_set(L, i, j, 0.0);
                }
            }
        }
    }
    
    // Printing the L matrix
    printf("\nL matrix:\n");
    print_matrix_row_wise(L);

    // Printing the U matrix
    printf("\nU matrix:\n");
    print_matrix_row_wise(U);

    // Computing the multiplication of L and U
    gsl_matrix *LU = gsl_matrix_alloc(A->size1, A->size1);
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, L, U, 0.0, LU);

    // Printing the result of multiplication (A = LU)
    printf("\nLU matrix after multiplication (A = LU):\n");
    print_matrix_row_wise(LU);

    // Free allocated memory
    gsl_permutation_free(p);
    gsl_matrix_free(L);
    gsl_matrix_free(U);
    gsl_matrix_free(LU);
}

int main(void)
{
    // Defining the coefficients of the linear equation as array
    double data1[] = { 3.0, -1.0, 1.0, 3.0, 6.0, 2.0, 3.0, 3.0, 7.0 };
    double data2[] = { 10.0, -1.0, 0.0, -1.0, 10.0, -2.0, 0.0, -2.0, 10.0 };
    double data3[] = { 10.0, 5.0, 0.0, 0.0, 5.0, 10.0, -4.0, 0.0, 0.0, -4.0, 8.0, -1.0, 0.0, 0.0, -1.0, 5.0 };
    double data4[] = { 4.0,1.0,1.0,0.0,1.0,-1.0,-3.0,1.0,1.0,0.0,2.0,1.0,5.0,-1.0,-1.0,-1.0,-1.0,-1.0,4.0,0.0,0.0,2.0,-1.0,1.0,4.0 };

    // Defining the data array as matrix
    gsl_matrix_view m1 = gsl_matrix_view_array(data1, 3, 3);
    gsl_matrix_view m2 = gsl_matrix_view_array(data2, 3, 3);
    gsl_matrix_view m3 = gsl_matrix_view_array(data3, 4, 4);
    gsl_matrix_view m4 = gsl_matrix_view_array(data4, 5, 5);

    // Performing LU decomposition and checking whether it is correct or not
    printf("\nMatrix M1:\n");
    print_matrix_row_wise(&m1.matrix);
    lu_decomposition(&m1.matrix);
    printf("\n--------------------------------------------------------------------------------------\n");
    printf("\nMatrix M2:\n");
    print_matrix_row_wise(&m2.matrix);
    lu_decomposition(&m2.matrix);
    printf("\n--------------------------------------------------------------------------------------\n");
    printf("\nMatrix M3:\n");
    print_matrix_row_wise(&m3.matrix);
    lu_decomposition(&m3.matrix);
    printf("\n--------------------------------------------------------------------------------------\n");
    printf("\nMatrix M4:\n");
    print_matrix_row_wise(&m4.matrix);
    lu_decomposition(&m4.matrix);
    printf("\n--------------------------------------------------------------------------------------\n");
    
    return 0;
}

