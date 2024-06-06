import numpy as np
import matplotlib.pyplot as plt

## a)
def build_wilk_bidiag(n: int):
    # init wilkenson bidiag
    A = np.zeros((n,n))

    for i in range(n):
        A[i,i] = n-i # diagonal principal: n, n-1, ..., 2, 1
        try:
            A[i,i+1] = n  # direita da diagonal principal: n, n, ..., n, n
        except:
            pass
    return A

# Teste
# print(build_wilk_bidiag(5))

## b)
outputs = []
for n in range(1,16):
    A = build_wilk_bidiag(n)
    # Use numpy to get cond number
    cond_number = np.linalg.cond( A )
    # Store for later graph
    outputs.append(cond_number)

# Plot
plt.plot(outputs, marker='o')
plt.xlabel("Wilkinson Bidiagonal Matrix Order")
plt.ylabel("Número de condição"); 
plt.show()

## c)
# Build order 20 winkenson matrix:
A = build_wilk_bidiag(20)
# Check its eigenvalues
eigvalues, eigvector = np.linalg.eig(A)
print("Autovalores Wikenson(20):\n",eigvalues) # 20, 19, ..., 2, 1 as expected
# Perturb element A(20,1) by 10^(-10)
A[19,0] += 10e-10
# Check its eigenvalues again
eigvalues, eigvector = np.linalg.eig(A)
print("Autovalores Wikenson(20) perturbada:\n",eigvalues) # 2 resultados reais e 18 complexos
