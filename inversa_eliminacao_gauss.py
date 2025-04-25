"""Implementação do método de eliminação de Gauss para encontrar a inversa de 
uma matriz quadrada. O método consiste em transformar a matriz original em uma 
matriz identidade, enquanto aplica as mesmas operações na matriz identidade, 
resultando na inversa da matriz original."""
import numpy as np

def inversa_eliminacao(A):
    """Recebe uma matriz quadrada A e retorna a inversa de A, caso exista."""
    A = np.array(A,dtype=float)
    n = len(A)
    for i in range(n):
        if int(A[i,i]) == 0:
            raise Exception("Elemento nulo encontrado na diagonal, sugiro trocar de método.")
    Ainv = np.eye(n)
    aumentada = [[] for i  in range(n)]
    for i in range(n):
        aumentada[i] = list(A[i]) +  list(Ainv[i])
    aumentada = np.array(aumentada)
    ## fazer um primero pivoteamento na diagonal principal
    for j in range(n):
        k = A[j,j]
        aumentada[j,:] = aumentada[j,:]/k
    for coluna in range(n):
        for j in range(n):
            # Fazer L1 = L1 - k*L0 de forma que A[1,0] = 0
            # k*L0 = L1 -> k = L1/L0 
            # fazer o mesmo pra todos as linhas: A[i,0] = 0 
            if coluna == j: continue
            k = aumentada[j,coluna]/aumentada[coluna,coluna]
            aumentada[j,:] = aumentada[j,:] - k*aumentada[coluna,:]
            # Essa linha deixou de ter pivot na diagonal principal
            # força ela a ter pivot novamente:
            k = aumentada[j,j]
            if -1e-10<= k <= 1e-10: 
                print(f'divisor={k}')
                raise Exception("Elemento nulo encontrado no pivoteamento da diagonal, matriz não invertível.")
            aumentada[j,:] = aumentada[j,:]/k
    Ainv = aumentada[:,n:]
    return Ainv

# Testes:
A = np.array([[1,2],
              [3,4]])
print(np.linalg.inv(A))
print(inversa_eliminacao(A))

print()
A = np.array([[2,3],
              [4,5]])
print(np.linalg.inv(A))
print(inversa_eliminacao(A))

print()
A = np.array([[2,3,0],
              [4,1,6],
              [1,2,8]])
print(np.linalg.inv(A))
print(inversa_eliminacao(A))

print()
A = np.array([[1,2,3],
              [4,5,6],
              [7,8,9]])
print(np.linalg.inv(A))
print(inversa_eliminacao(A))