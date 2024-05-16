import numpy as np

def inversa_eliminacao(A):
    A = np.array(A,dtype=float)
    n = len(A)
    for i in range(n):
        if int(A[i,i]) == 0:
            Exception("Elemento nulo encontrado na diagonal, sugiro trocar de método.")
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