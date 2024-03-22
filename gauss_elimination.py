""" This program calculates the solution of four linear equation using numpy.linalg.solve()"""

import numpy as np

"""
For part (a)
   3x1 −  x2 +  x3 = 1
   3x1 + 6x2 + 2x3 = 0
   3x1 + 3x2 + 7x3 = 4
"""
A = np.array([[3, -1, 1], [3, 6, 2], [3, 3, 7]])
b = np.array([1, 0, 4])

solution = np.linalg.solve(A, b)
print("Equations:\n3x1 − x2 + x3 = 1\n3x1 + 6x2 + 2x3 = 0\n3x1 + 3x2 + 7x3 = 4")
print("Solution:", solution, "\n")  # Printing solution

"""
For part (b)
    10x1 −   x2        = 9
    − x1 + 10x2 −  2x3 = 7
         −  2x2 + 10x3 = 6
"""

A = np.array([[10, -1, 0], [-1, 10, -2], [0, -2, 10]])
b = np.array([9, 7, 6])

solution = np.linalg.solve(A, b)
print("Equations:\n10x1 − x2 = 9\n−x1 + 10x2 − 2x3 = 7\n−2x2 + 10x3 = 6")
print("Solution:", solution, "\n")  # Printing solution

"""
For part (c)
    10x1 + 5x2              = 6
     5x1 + 10x2 − 4x3       = 25
         −  4x2 + 8x3 −  x4 = −11
                −  x3 + 5x4 = −11
"""

A = np.array([[10, 5, 0, 0], [5, 10, -4, 0], [0, -4, 8, -1], [0, 0, -1, 5]])
b = np.array([6, 25, -11, -11])

solution = np.linalg.solve(A, b)
print("Equations:\n10x1 + 5x2 = 6\n5x1 + 10x2 − 4x3 = 25\n− 4x2 + 8x3 −  x4 = −11\n− x3 + 5x4 = −11")
print("Solution:", solution, "\n")  # Printing solution

"""
For part (d)
4x1 +  x2 +  x3        +  x5 = 6
−x1 − 3x2 +  x3 +   x4       = 6
2x1 +  x2 + 5x3 −   x4 −  x5 = 6
−x1 − x2 −   x3 +  4x4       = 6
     2x2 −   x3  +  x4 + 4x5 = 6
"""

A = np.array([[4, 1, 1, 0, 1], [-1, -3, 1, 1, 0], [2, 1, 5, -1, -1], [-1, -1, -1, 4, 0], [0, 2, -1, 1, 4]])
b = np.array([6, 6, 6, 6, 6])

solution = np.linalg.solve(A, b)
print("Equations:\n4x1 + x2 + x3 + x5 = 6\n−x1 − 3x2 + x3 + x4 = 6\n2x1 + x2 + 5x3 − x4 − x5 = 6\n−x1 − x2 −x3 + 4x4 = 6")
print("2x2 − x3 + x4 + 4x5 = 6")
print("Solution:", solution, "\n")  # Printing solution

