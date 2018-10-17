import numpy as np

def SSOR(A,b,e,w,x):
    r = np.dot(A,x) - b
    n = b.size
    D = np.diag(np.diag(A))
    L = np.tril(A)
    Q = np.dot(np.dot((D + w * L),np.linalg.inv(D)),np.transpose(D + w * L))
    z = np.dot(np.linalg.inv(Q),r)
    p = -(np.copy(z))
    while(np.linalg.norm((np.dot(A, x) - b), ord=2)>e):
        q = np.dot(A,p)
        a = np.dot(np.transpose(r),z)[0][0]/(np.dot(np.transpose(p),q)[0][0])
        x = x + a * p
        beta = 1 / (np.dot(np.transpose(r),z)[0][0])
        r = r + a * q
        z = np.dot(np.linalg.inv(Q),r)
        beta = beta * np.dot(np.transpose(r),z)[0][0]
        p = -z + beta * p
    return x

import Defaul_Matrix

A = Defaul_Matrix.Default_matrix(40)
b = np.dot(A,np.array([[1] for i in range(1600)],dtype='float64'))
x = np.array([[0] for i in range(1600)],dtype='float64')
Res = SSOR(A,b,1e-6,0,x)
print(Res)