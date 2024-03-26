# This code calculates the QR decompositon of a matrix and hence finds its eigenvalues and compares it with that obtained from numpy.linalg.eigh

import numpy as np

# Defining the matrix A
A = np.array([[5, -2],
              [-2, 8]])

# QR decomposition of A for a number of iterations
Ak = np.copy(A)
iterations = 30
for i in range(iterations):
    Q, R = np.linalg.qr(Ak)
    Ak = Q.T @ Ak @ Q
    if i == 0:
      print("\nQR decomposition of A:\nQ = ", Q, "\n\nR =", R)


# Calculating eigenvalues using QR decomposition
eigenvalues_QR = np.diag(Ak)
# Calculating eigenvalues using numpy.linalg.eigh
eigenvalues_eigh = np.linalg.eigh(A)[0]

# Printing the results
print("\nEigenvalues using QR decomposition:", eigenvalues_QR)
print("Eigenvalues using numpy.linalg.eigh:", eigenvalues_eigh)
import numpy as np

