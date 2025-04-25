"""Implementa a retrosubstituição (back_subs), um método que calcula a solução 
de um sistema linear triangular superior de baixo para cima."""

import numpy as np

def back_subs(A,b):
	"""Recebe uma matriz triangular superior A e um vetor b, que representam um
	sistema linear Ax = b, da qual deseja-se retornar a solução x."""
	n, _ = A.shape
	x = [0 for i in range(n)]
	for i in range(n-1,-1,-1):
		# Método normal
		# soma = 0
		# for j in range(i,n):
		# 	soma += A[i,j]*x[j]
		# xi = (b[i] - soma)/A[i,i]
		# x[i] = xi

		# Método one line
		x[i] = (b[i] - sum([A[i,j]*x[j] for j in range(i,n)]))/A[i,i]
	return x

## Exemplo de uso:

# Sistema S:
# 1.x1 + 1.x2 - 1.x3 = 2
# 2.x1 + 1.x2 + 1.x3 = 0
# -1.x1- 2.x2 + 3.x3 = 4

## Em triangular sup
# eq1            : 1.x1 + 1.x2 - 1.x3 = 2
# eq2 = eq2-2*eq1: 0.x1 -1.x1 + 3.x3 = -4 
# eq3 = eq3+eq1  : 0.x1 -1.x1 + 2.x3 = 6
# eq3 = eq3-1*eq2: 0.x1 +0.x1 - 1.x3 = 10

# Sistema S em triangular sup:
# 1.x1 + 1.x2 - 1.x3 = 2
# 0.x1 -1.x1 + 3.x3 = -4 
# 0.x1 +0.x1 - 1.x3 = 10

A = np.array([[1,1,-1],
              [0,-1,3],
              [0,0,-1]])
b = [2,-4, 10]

print(back_subs(A,b))
