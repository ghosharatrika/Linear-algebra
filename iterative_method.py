#This code calculates the solution of a linear equation using jacobi, gauss seidal, relaxation(w=1.25) and conjugate method with a tolerance of 0.01

import numpy as np

# Given matrix A and vector b
A = np.array([[0.2, 0.1, 1, 1, 0], [0.1, 4, -1, 1, -1], [1, -1, 60, 0, -2], [1, 1, 0, 8, 4],
              [0, -1, -2, 4, 700]])
b = np.array([1., 2., 3., 4., 5.])

# True solution vector
true_solution = np.array([7.859713071, 0.422926408, -0.073592239, -0.540643016, 0.010626163])

tolerance = 0.01  # Tolerance
n = len(b)


# Jacobi method
def jacobi_method(A, b, tolerance):
    x = np.zeros_like(b)  # Initial guess
    iterations = 0
    r = A - np.diag(np.diagonal(A))
    while True:
        x_new = (b - np.dot(r, x))/np.diagonal(A)
        iterations += 1
        if np.linalg.norm(x_new - x) < tolerance:
            break
        x = x_new
    return x, iterations


# Gauss-Seidel method
def gauss_seidel_method(A, b, tolerance):
    x = np.zeros_like(b)  # Initial guess
    iterations = 0
    while True:
        x_new = np.copy(x)
        for i in range(n):
            x_new[i] = (b[i] - np.dot(A[i, :i], x_new[:i]) - np.dot(A[i, i + 1:], x_new[i + 1:])) / A[i, i]
        iterations += 1
        if np.linalg.norm(x_new - x) < tolerance:
            break
        x = x_new
    return x, iterations


# Relaxation method
def relaxation_method(A, b, tolerance, w):
    x = np.zeros_like(b)  # Initial guess
    iterations = 0
    while True:
        x_new = np.copy(x)
        for i in range(n):
            x_new[i] = ((1 - w) * x[i] + w * (b[i] - np.dot(A[i, :i], x_new[:i]) - np.dot(A[i, i + 1:], x[i + 1:])) / A[i, i])
        iterations += 1
        if np.linalg.norm(x_new - x) < tolerance:
            break
        x = x_new
    return x, iterations


# Conjugate Gradient method
def conjugate_gradient_method(A, b, tolerance):
    x = np.zeros_like(b)  # Initial guess
    r = b - np.dot(A, x)
    p = np.copy(r)
    iterations = 0
    while True:
        alpha = np.dot(r, r) / np.dot(np.dot(p, A), p)
        x_new = x + alpha * p
        r_new = r - alpha * np.dot(A, p)
        beta = np.dot(r_new, r_new) / np.dot(r, r)
        p = r_new + beta * p
        iterations += 1
        if np.linalg.norm(x_new - x) < tolerance:
            break
        x = x_new
        r = r_new
    return x, iterations


# Solving using each method
jacobi_solution, jacobi_iterations = jacobi_method(A, b, tolerance)
gauss_seidel_solution, gauss_seidel_iterations = gauss_seidel_method(A, b, tolerance)
relaxation_solution, relaxation_iterations = relaxation_method(A, b, tolerance, w=1.25)
cg_solution, cg_iterations = conjugate_gradient_method(A, b, tolerance)

# Printing solutions and iterations
print("Jacobi method:")
print("Solution:", jacobi_solution)
print("Iterations:", jacobi_iterations, "\n")
print("Gauss-Seidel method:")
print("Solution:", gauss_seidel_solution)
print("Iterations:", gauss_seidel_iterations, "\n")
print("Relaxation method:")
print("Solution:", relaxation_solution)
print("Iterations:", relaxation_iterations, "\n")
print("Conjugate Gradient method:")
print("Solution:", cg_solution)
print("Iterations:", cg_iterations, "\n")
