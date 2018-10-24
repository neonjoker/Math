import numpy as np
from Defaul_Matrix import Default_matrix
import csv

def SOR(A,b,e,w=1,Stop_Method=0):
    n = b.size
    x = np.array([[0] for i in range(n)],dtype='float64')
    count = 0
    SOROut = open('3-SOR.csv', 'w', newline='')
    Out = csv.writer(SOROut, dialect='excel')
    def Move_Forward():
        nonlocal x
        nonlocal count
        for i in range(n):
            tmp_x = x[i][0]
            x[i][0] = b[i][0]
            for j in range(n):
                if (j != i):
                    x[i][0] = x[i][0] - A[i, j] * x[j][0]
            x[i][0] = (1-w)* tmp_x + w * x[i][0] / A[i, i]
        x = x.copy()
        count = count + 1
        Out.writerow([count,np.linalg.norm(np.array([[1] for i in range(n)]) - x, ord=2), np.linalg.norm((np.dot(A, x) - b), ord=2)])
    tmp_1 = x.copy()
    Move_Forward()
    if(Stop_Method != 2):
        if(Stop_Method == 0):
            def r():
                rest = np.linalg.norm((np.dot(A, x) - b), ord=2)
                return rest
            def solve():
                while(abs(r())>e and count <= 1000):
                    Move_Forward()
            solve()
        elif(Stop_Method == 1):
            def r(x1,x2):
                return np.linalg.norm((x1-x2),ord = 2)
            def solve():
                nonlocal tmp_1
                while(r(x,tmp_1)>e and count <= 1000):
                    tmp_1 = x.copy()
                    Move_Forward()
            solve()
    elif(Stop_Method == 2):
        def r(x1,x2,x3):
            d1 = np.linalg.norm((x1-x2), ord = 2)
            d2 = np.linalg.norm((x2-x3), ord = 2)
            return (d2*d2/(d1-d2))
        def solve():
            Move_Forward()
            nonlocal tmp_1
            tmp_2 = tmp_1.copy()
            tmp_1 = x.copy()
            while(r(x,tmp_1,tmp_2)<e and count <= 1000):
                Move_Forward()
                tmp_2 = tmp_1.copy()
                tmp_1 = x.copy()
        solve()
    return (x,count)

'''Test of SOR'''
'''
SOROut = open('SOR.csv','w',newline='')
Out = csv.writer(SOROut,dialect='excel')
for i in np.linspace(0.0,2.0,num=20):
    A = Default_matrix(10)
    b = np.array([[1] for i in range(100)],dtype='float64')
    x = SOR(A,b,1e-6,i)
    Out.writerow([i,x[1]])


A = Default_matrix(10)
b = np.array([[1] for i in range(100)],dtype='float64')
x = SOR(A,b,1e-6,1.57894736842105)
'''

SOROut = open('32-Sor.csv','w',newline='')
Out = csv.writer(SOROut, dialect='excel')
for j in range(3,11):
    A = Default_matrix(j)
    b = np.dot(A, np.array([[1] for i in range(j * j)], dtype='float64'))
    x = SOR(A,b,1e-6,1.57894736842105,0)[1]
    Out.writerow([j,x])