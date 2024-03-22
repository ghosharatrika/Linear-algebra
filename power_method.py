# This code calculates the dominant eigenvalue and corresponding eigenvector of a matrix using power method

import numpy as np

# Defining the matrix A
A = np.array([[2, -1, 0],
              [-1, 2, -1],
              [0, -1, 2]])

# Initial guess for the eigenvector
x = np.ones(A.shape[0])  # x = [1, 1, 1]
y = np.array([1, 1, 1])  # y = [1, 1, 1] 
tolerance = 1e-2  # 1 percent tolerance

# Power Method
x = A @ x
eigen_val = np.dot(A @ x, y) / np.dot(x, y)

while True:
    x = A @ x
    eigen_val1 = np.dot(A @ x, y) / np.dot(x, y)  # Updating the eigenvalue approximation
    x = x / np.linalg.norm(x)  # Updating the eigenvector approximation

    # Checking for convergence
    if np.abs(eigen_val - eigen_val1) < tolerance:
        break
    eigen_val = eigen_val1

# Printing the results
print("Dominant eigenvalue:", eigen_val)
print("Dominant eigenvector:", x)
