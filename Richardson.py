import numpy as np


def Richardson(A,b,x,m):
    n = b.size
    EigMax = 4 * (1-np.cos((np.sqrt(n)*np.pi)/(np.sqrt(n)+1)))
    EigMin = 4 * (1 - np.cos((np.pi) / (np.sqrt(n) + 1)))
    G = np.diag(np.diag(np.ones_like(A))) - (2 / (EigMax + EigMin)) * A
    g = (2 / (EigMin + EigMax)) * b
    tmp_x = 2 / (EigMax + EigMin) * (b - np.dot(A, x))
    Eig = np.linalg.eigvals(G)
    EigMax = np.max(Eig)
    EigMin = np.min(Eig)
    ksi = (2 - EigMax - EigMin)/(EigMax - EigMin)
    rho = 2.0
    v = 2 / (2 - EigMax - EigMin)
    for i in range(m):
        rho = 1 / (1 - (rho / (4 * ksi * ksi)))
        tmp = (1 - rho) * x + rho * (v * (np.dot(G,tmp_x) + g) + (1 - v) * tmp_x)
        x = np.copy(tmp_x)
        tmp_x = np.copy(tmp)
    return (tmp_x,x)


def r(A,b,x):
    rest = np.linalg.norm((np.dot(A, x) - b), ord=2)
    return rest

import Defaul_Matrix
A = Defaul_Matrix.Default_matrix(3)
b = np.dot(A,np.array([[1] for i in range(9)],dtype='float64'))
x = np.array([[0] for i in range(9)],dtype='float64')
while(r(A,b,x)>1e-6):
    Res = Richardson(A,b,x,20)
    x = Res[0]
print(x)