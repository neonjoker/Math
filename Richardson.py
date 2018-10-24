import numpy as np


def Richardson(A,b,x,m):
    n = b.size
    EigMax = 4 * (1-np.cos((np.sqrt(n)*np.pi)/(np.sqrt(n)+1)))
    EigMin = 4 * (1 - np.cos((np.pi) / (np.sqrt(n) + 1)))
    for i in range(m):
        t = 1 / ((EigMax-EigMin)/2 * np.cos((2* i + 1) * np.pi / 2 * m) + (EigMin + EigMax)/2)
        x = x + t * (b - np.dot(A,x))
    return x


def r(x):
    n = x.size
    rest = np.linalg.norm(np.array([[1] for i in range(n)]) - x, ord=2)
    return rest

def res(A,b,x):
    rest = np.linalg.norm((np.dot(A, x) - b), ord=2)
    return rest

import Defaul_Matrix
import csv
def testOfR(m):
    name = 'Richardson_' + str(m) + '.csv'
    RichardsonOut = open(name,'w',newline='')
    Out = csv.writer(RichardsonOut,dialect='excel')
    A = Defaul_Matrix.Default_matrix(20)
    b = np.dot(A,np.array([[1] for i in range(400)],dtype='float64'))
    x = np.array([[0] for i in range(400)],dtype='float64')
    count = 0
    while(r(x)>1e-6 and count<=1000):
        Res = Richardson(A,b,x,m)
        x = Res
        count = count + 1
        Out.writerow([r(x),res(A,b,x)])
'''
for m in [5,6,7,8,9,10,15,20]:
    testOfR(m)
'''
testOfR(6)
'''
print(Richardson(Defaul_Matrix.Default_matrix(3),np.dot(Defaul_Matrix.Default_matrix(3),np.array([[1] for i in range(9)],dtype='float64')),np.array([[0] for i in range(9)],dtype='float64'),5))
'''