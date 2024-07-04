import numpy as np

def alc_cholesky(A):
    # Factor the positive definite matrix A
    # using the Cholesky decomposition algorithm.
    # If the algorithm fails, A is not positive definite.
    # Output an error message and return and empty array R.
    n = len(A)
    R = np.zeros((n,n))
    for i in range(n):
        soma = 0
        for j in range(i):
            soma += R[j,i]*R[j,i]
        tmp = A[i,i] - soma
        if tmp <= 0:
            raise Exception("A entrada não é positiva definida, impossibilidade de realizar a decomposição de Cholesky")
            return []
        R[i,i] = tmp**(0.5)
        for j in range(i+1,n):
            soma2 = 0
            for k in range(i):
                soma2 += R[k,i]*R[k,j]
            R[i,j] = (A[i,j] - soma2)/R[i,i]
    return R

### Testes
print("Algorithm 13.3 The Cholesky Decomposition - FORD: ")

A = [[1 , 1, 4,-1],
     [1 , 5, 0,-1],
     [4 , 0,21,-4],
     [-1,-1,-4,10]]
A = np.array(A)
R = alc_cholesky(A)
print("A=R^T @ R ?", np.allclose(R.T @ R, A))

# print("Cholesky Decomposition via np.linalg.cholesky: ")
# L = np.linalg.cholesky(A)
# print("A=L @ L^T ?", np.allclose(L @ L.T, A))

## test error messga
# print("\nTest error message:")
# B = [[1 ,5 , 6],
#      [-7,12, 5],
#      [2 ,1 ,10]]
# B = np.array(B)
# R = alc_cholesky(B)
# if R:
#     print("B=R^T @ R ?", np.allclose(R.T @ R, B))
