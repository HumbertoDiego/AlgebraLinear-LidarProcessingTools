import numpy as np

P1 = [[-0.40825 , 0.43644 , 0.80178 ],
      [-0.8165  ,0.21822  ,-0.53452],
      [-0.40825 ,-0.87287 ,0.26726]]

P2 = [[-0.51450 ,0.48507  ,0.70711],
      [-0.68599 ,-0.72761 ,0.0000],
      [0.51450  ,-0.48507 ,0.70711]]

"""
P1 = [[-0.58835 , 0.70206 , 0.40119],
      [-0.78446 , -0.37524, -0.49377] ,
      [-0.19612 , -0.60523, 0.77152]]

P2 = [[-0.47624 , -0.4264  ,0.30151] ,
      [0.087932 , 0.86603  ,-0.40825],
      [-0.87491 , -0.26112 ,0.86164]]
"""

def is_orthogonal_by_definition(P, tol = 1e-04):
    P = np.array(P)
    if P.shape[0]==P.shape[1]:
        n = P.shape[0]
        I = np.eye(n)
        # P.T @ P == I?
        # print(P.T @ P == I) # proximos mas diferentes
        is_close = np.isclose(P.T @ P, I, atol = tol)
        if is_close.all():
            flag = True
        else:
            flag = False
    else:
        flag = False
    print("is_orthogonal_by_definition?",flag)
    return flag

def is_orthogonal_by_vectors(P, tol = 1e-04):
    P = np.array(P)
    if P.shape[0]==P.shape[1]:
        n = P.shape[0]
        ones = np.ones(n)
        colunas = [ P[:,i] for i in range(n) ]
        # columns have unit length ?
        lenghts = [ np.linalg.norm(c) for c in colunas]
        # lenghts == [1. ,1. ,1. ]? # proximos mas diferentes
        is_close = np.isclose(lenghts, ones, atol = tol)
        if is_close.all():
            # columns are orthogonal to each other ?
            orthogonals = []
            computeds = []
            for i in range(n):
                for j in range(n):
                    if i==j:continue
                    if (i,j) in computeds or (j,i) in computeds: continue
                    prod_int = colunas[i] @ colunas[j]
                    orthogonals.append(np.isclose(prod_int, 0, atol=tol))
                    computeds.append( (i,j) )
            orthogonals = np.array(orthogonals, dtype= bool)
            if orthogonals.all():
                flag = True
            else:
                flag = False
        else:
            flag = False
    else:
        flag = False
    print("is_orthogonal_by_vectors?",flag)
    return flag

is_orthogonal_by_definition(P1)
is_orthogonal_by_vectors(P1)
is_orthogonal_by_definition(P2)
is_orthogonal_by_vectors(P2)