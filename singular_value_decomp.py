# To calculate the singular value decompositon and demonstrate its correctness

import numpy as np
import time

# Defining the matrices
matrix1 = np.array([[2, 1], [1, 0]], dtype=float)
matrix2 = np.array([[2, 1, 0], [1, 0, 1]], dtype=float)
matrix3 = np.array([[2, 1], [-1, 1], [1, 1], [2, -1]], dtype=float)
matrix4 = np.array([[1, 1, 0], [-1, 0, 1], [0, 1, -1], [1, 1, -1]], dtype=float)
matrix5 = np.array([[0, 1, 1], [0, 1, 0], [1, 1, 0], [0, 1, 0], [1, 0, 1]], dtype=float)


def svd_matrix(matrix):  # Computing SVD for matrix1
    start_time = time.time()
    U, S1, V_Transpose = np.linalg.svd(matrix)
    
    """ numpy.linalg.svd gives only the nonzero elements of S matrix which is stored in S1 array.
    Hence to get S matrix, first defining it to be of same dimension as the original matrix and
    then inputing S1 array in the "diagonals" of S matrix
    """
    
    S = np.zeros_like(matrix)
    dimension = len(S1)
    for i in range(dimension):
        S[i][i] = S1[i]
    end_time = time.time()
    print("\nSVD for matrix:\n", matrix)
    print("Time taken for calculating SVD:", end_time - start_time)  # To report the computation time
    # Printing the U, S, V transpose matrix
    print("U:\n", U)
    print("S:\n", S)
    print("V_Transpose:\n", V_Transpose)
    # Demonstrating correctness of decomposition for matrix as A = U @ S @ V_Transpose
    print("\nDemonstrating correctness of decomposition for matrix:")
    reconstructed_matrix1 = U.dot(S).dot(V_Transpose)
    print("Reconstructed Matrix:")
    print(reconstructed_matrix1)


svd_matrix(matrix1)
svd_matrix(matrix2)
svd_matrix(matrix3)
svd_matrix(matrix4)
svd_matrix(matrix5)
