import numpy as np
from scipy.linalg import lu

def ludecomp(A):
    """
    LU decomposition using Gaussian elimination with partial pivoting. 
    [P U P interchanges] = ludecomp(A) factors a square
    matrix so that PA = LU. U is an upper-triangular matrix,
    L is a lower-triangular matrix, and P is a permutation
    matrix that reflects the row exchanges required by
    partial pivoting used to reduce round-off error.
    In the event that is useful, interchanges is the number
    of row interchanges required.
    """
    A = A.astype(float)
    n = A.shape[0]
    L = np.eye(n)
    P = np.eye(n)
    for i in range(n-1):
        column_i = A[:,i]
        rows_i_to_n = column_i[i:n]
        k = np.argmax(rows_i_to_n) # index of largest matrix entry in column i, rows i through n
        pivotindex = i + k
        if pivotindex != i:
            # Exchange rows i and k, ignoring columns 1 through i-1 in each row.
            tmp = A[i, i:n].copy()
            A[i, i:n] = A[pivotindex, i:n]
            A[pivotindex, i:n] = tmp
            
            # Swap whole rows in P.
            tmp = P[i, :].copy()
            P[i, :] = P[pivotindex, :]
            P[pivotindex, :] = tmp

            # Swap rows of L also, but only in columns 1 through i-1.
            tmp = L[i, 0:i].copy()
            L[i, 0:i] = L [pivotindex ,0:i]
            L[pivotindex, 0:i] = tmp
        
        # Compute the multipliers.
        multipliers = A[i+1:n ,i]/A[i, i]

        # Use submatrix calculations instead of a loop to perform
        # the row operations on the submatrix A(i+1:n, i+1:n)
        A[i+1:n, i+1:n] -= np.outer(multipliers, A[i, i+1:n])
                                               
        # Set entries in column i, rows i+1:n to 0.
        A[i+1:n, i] = 0
        L[i+1:n, i] = multipliers
    U = A.copy()
    U[np.where(np.abs(U)<10e-5)] = 0 # float number imprecision correction
    return L,U,P

B = np.array([[1, 2, 3],
              [2, 5, 4],
              [3, 5, 4]])

L, U, P = ludecomp(B)

print("Implementação Algorithm 11.2 Gaussian Elimination with Partial Pivoting: ")
print("L=\n",L,"\nU=\n",U,"\nP=\n",P)
print("PB=\n",P @ B)
print("LU=\n",L @ U)
print("PB=LU ?", np.allclose( P @ B , L @ U ) )

print("\n\nLU factorization via scipy.linalg.lu: ")
P, L, U = lu(B)
U[np.where(np.abs(U)<10e-5)] = 0 # float number imprecision correction

print("L=\n",L,"\nU=\n",U,"\nP=\n",P)
print("B=\n",B)
plu = P @ L @ U
print("PLU=\n", plu)
print("B=PLU ?", np.allclose( B , plu ) )

