"""Implementação do produto vetorial."""
import numpy as np

def crossprod(u,v):
	# Em termos das componentes unitárias canônicas i, j e k:
    # u × v = i(u2v3 − u3v2) − j(u1v3 − u3v1) + k(u1v2 − u2v1)
	prod = [ u[1]*v[2] - u[2]*v[1],
		    -u[0]*v[2] + u[2]*v[0],
			 u[0]*v[1] - u[1]*v[0] 
            ]
	return np.array(prod)

# Testes
u = np.array([0 , 1 , 2])
v = np.array([3 , 4 , 5])

print("u x v =",crossprod(u,v))
print("v x u =", crossprod(v,u))
print("<u x v , u> =", np.inner(crossprod(u,v), u) )
print("<v x u , v> =", np.inner(crossprod(v,u), v) )