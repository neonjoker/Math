import numpy as np

def CG(A,b,e,x):
    r = np.dot(A,x) - b
    p = -(np.copy(r))
    while(np.linalg.norm((np.dot(A, x) - b), ord=2)>e):
        q = np.dot(A,p)
        a = np.dot(np.transpose(r),r)[0][0]/(np.dot(np.transpose(p),q)[0][0])
        x = x + a * p
        beta = 1 / (np.dot(np.transpose(r),r)[0][0])
        r = r + a * q
        beta = beta * np.dot(np.transpose(r),r)[0][0]
        p = -r + beta * p
    return x

import Defaul_Matrix

A = Defaul_Matrix.Default_matrix(3)
b = np.dot(A,np.array([[1] for i in range(9)],dtype='float64'))
x = np.array([[0] for i in range(9)],dtype='float64')
Res = CG(A,b,1e-6,x)
print(Res)