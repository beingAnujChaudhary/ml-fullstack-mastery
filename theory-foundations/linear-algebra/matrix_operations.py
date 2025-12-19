"""
Matrix operations implemented from scratch using Python lists.
"""

from typing import List

Matrix = List[List[float]]


def shape(M: Matrix):
    return len(M), len(M[0])


def matmul(A: Matrix, B: Matrix) -> Matrix:
    rows_A, cols_A = shape(A)
    rows_B, cols_B = shape(B)

    assert cols_A == rows_B, "Invalid matrix dimensions"

    result = [[0.0 for _ in range(cols_B)] for _ in range(rows_A)]

    for i in range(rows_A):
        for j in range(cols_B):
            for k in range(cols_A):
                result[i][j] += A[i][k] * B[k][j]

    return result


if __name__ == "__main__":
    A = [[1, 2], [3, 4]]
    B = [[5, 6], [7, 8]]

    for row in matmul(A, B):
        print(row)
