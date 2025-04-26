"""Modified Gram-Schmidt QR Decomposition Implementation
"""
import numpy as np

def modqrgrsch(A):
    """Input: m x n matrix A.
    Output: the QR decomposition A = QR, where
    Q is an m x n matrix with orthonormal columns, and 
    R is an n x n upper-triangular matrix."""
    A = A.astype(float)
    m,n = A.shape
    Q = np.zeros((m,n))
    R = np.zeros((n,n))
    for i in range(n):
        Q[:,i] = A[:,i]
        for j in range(i):
            R[j, i] = np.dot(Q[:, j], Q[:, i])
            Q[:, i] = Q[:, i] - R[j, i] * Q[:, j]
        R[i, i] = np.linalg.norm(Q[:, i ])
        Q[:, i] = Q[:, i]/R[i, i] 
    return Q,R

A = np.array ([[1, 9 ,0 ,5 ,3 ,2],
               [-6,3 ,8 ,2 ,-8,0],
               [3, 15,23,2 ,1 ,7],
               [3, 57,35,1 ,7 ,9],
               [3, 5 ,6 ,15,55,2],
               [33,7 ,5 ,3 ,5 ,7]])

print("Algorithm 14.3 Modified Gram-Schmidt QR Decomposition: ")
Q,R = modqrgrsch(A)
print("Q=\n",Q,"\nR=\n",R)
qr = Q @ R
qr[ np.where( np.abs(qr) < 10e-5 ) ] = 0 # float number imprecision correction
print("QR=\n", qr )
print("A=QR ?", np.allclose(A, qr) )

print("\n\nQR factorization via np.linalg.qr: ")
Q, R = np.linalg.qr(A)
print("Q=\n", Q, "\nR=\n", R)
qr = Q @ R
qr[np.where(np.abs(qr)<10e-5)] = 0 # float number imprecision correction
print("QR=\n", qr)
print("A=QR ?", np.allclose(A, qr) )