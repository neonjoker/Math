import numpy as np
import csv


def CG(A,b,e,x):
    r = np.dot(A,x) - b
    p = -(np.copy(r))
    n = b.size
    name = 'CG_' + str(np.sqrt(n)) + '.csv'
    CGOut = open(name, 'w', newline='')
    Out = csv.writer(CGOut, dialect='excel')
    count = 0
    while(np.linalg.norm((np.dot(A, x) - b), ord=2)>e):
        q = np.dot(A,p)
        a = np.dot(np.transpose(r),r)[0][0]/(np.dot(np.transpose(p),q)[0][0])
        x = x + a * p
        beta = 1 / (np.dot(np.transpose(r),r)[0][0])
        r = r + a * q
        beta = beta * np.dot(np.transpose(r),r)[0][0]
        p = -r + beta * p
        count = count + 1
        Out.writerow([np.linalg.norm(np.array([[1] for i in range(n)]) - x, ord=2),np.linalg.norm((np.dot(A, x) - b), ord=2)])
    return (x,count)

import Defaul_Matrix
A = Defaul_Matrix.Default_matrix(31)
b = np.dot(A,np.array([[1] for i in range(31*31)],dtype='float64'))
x = np.array([[0] for i in range(31*31)],dtype='float64')
Res = CG(A,b,1e-6,x)
print(Res[0])

A = Defaul_Matrix.Default_matrix(32)
b = np.dot(A,np.array([[1] for i in range(32*32)],dtype='float64'))
x = np.array([[0] for i in range(1024)],dtype='float64')
Res = CG(A,b,1e-6,x)
print(Res[0])

CGOut = open('CG', 'w', newline='')
Out = csv.writer(CGOut, dialect='excel')
for i in range(3,21):
    A = Defaul_Matrix.Default_matrix(i)
    b = np.dot(A, np.array([[1] for j in range(i * i)], dtype='float64'))
    x = np.array([[0] for j in range(i * i)], dtype='float64')
    Res = CG(A, b, 1e-6, x)
    Out.writerow([i+1,Res[1]])