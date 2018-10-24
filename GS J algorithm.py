import numpy as np
import Defaul_Matrix
import csv

def Jacobi_Method(A,b,e,Stop_Method):
    n = b.size
    x = np.array([[0] for i in range(n)],dtype='float64')
    count = 0
    res = 0
    JacobiOut = open('3-Jacobi.csv', 'w', newline='')
    Out = csv.writer(JacobiOut, dialect='excel')
    def Move_Forward():
        nonlocal x
        nonlocal count
        tmp_x = b.copy()
        for i in range(n):
            for j in range(n):
                if (j != i):
                    tmp_x[i][0] = tmp_x[i][0] - A[i, j] * x[j][0]
            tmp_x[i][0] = tmp_x[i][0] / A[i, i]
        x = tmp_x.copy()
        count = count + 1
        Out.writerow([count,np.linalg.norm(np.array([[1] for i in range(n)]) - x, ord=2), np.linalg.norm((np.dot(A, x) - b), ord=2)])
    tmp_1 = x.copy()
    Move_Forward()
    if(Stop_Method != 2):
        if(Stop_Method == 0):
            def r():
                rest = np.linalg.norm(np.array([[1] for i in range(n)]) - x, ord=2)
                return rest
            def solve():
                nonlocal res
                while(abs(r())>e):
                    Move_Forward()
                res = r()
            solve()
        elif(Stop_Method == 1):
            def r(x1,x2):
                return np.linalg.norm((x1-x2),ord = 2)
            def solve():
                nonlocal tmp_1
                nonlocal res
                while(r(x,tmp_1)>e):
                    tmp_1 = x.copy()
                    Move_Forward()
                res = r(x,tmp_1)
            solve()
    elif(Stop_Method == 2):
        def r(x1,x2,x3):
            d1 = np.linalg.norm((x1-x2), ord = 2)
            d2 = np.linalg.norm((x2-x3), ord = 2)
            return (d2*d2/(d1-d2))
        def solve():
            nonlocal tmp_1
            nonlocal res
            tmp_2 = x.copy()
            Move_Forward()
            while (r(tmp_1, tmp_2, x) > e):
                tmp_1 = tmp_2.copy()
                tmp_2 = x.copy()
                Move_Forward()
            res = r(tmp_1, tmp_2, x)
        solve()
    res = np.linalg.norm(np.array([[1] for i in range(n)]) - x, ord=2)
    return (x,count,res)

'''Test of Jacobi Algorithm'''
'''
A = Defaul_Matrix.Default_matrix(3)
b = np.array([[1] for i in range(9)],dtype='float64')
x = Jacobi_Method(A,b,1e-6,1)
print(x)
'''

def Gauss_Method(A,b,e,Stop_Method):
    n = b.size
    x = np.array([[0] for i in range(n)],dtype='float64')
    count = 0
    res = 0
    GaussOut = open('3-Gauss.csv', 'w', newline='')
    Out = csv.writer(GaussOut, dialect='excel')
    def Move_Forward():
        nonlocal x
        nonlocal count
        for i in range(n):
            x[i][0] = b[i][0]
            for j in range(n):
                if (j != i):
                    x[i][0] = x[i][0] - A[i, j] * x[j][0]
            x[i][0] = x[i][0] / A[i, i]
        x = x.copy()
        count = count + 1
        Out.writerow([count,np.linalg.norm(np.array([[1] for i in range(n)]) - x, ord=2), np.linalg.norm((np.dot(A, x) - b), ord=2)])
    tmp_1 = x.copy()
    Move_Forward()
    if(Stop_Method != 2):
        if(Stop_Method == 0):
            def r():
                rest = np.linalg.norm(np.array([[1] for i in range(n)]) - x, ord=2)
                return rest
            def solve():
                nonlocal res
                while(abs(r())>e):
                    Move_Forward()
                res = r()
            solve()
        elif(Stop_Method == 1):
            def r(x1,x2):
                return np.linalg.norm((x1-x2),ord = 2)
            def solve():
                nonlocal tmp_1
                nonlocal res
                while(r(x,tmp_1)>e):
                    tmp_1 = x.copy()
                    Move_Forward()
                res = r(x,tmp_1)
            solve()
    elif(Stop_Method == 2):
        def r(x1,x2,x3):
            d1 = np.linalg.norm((x1-x2), ord = 2)
            d2 = np.linalg.norm((x2-x3), ord = 2)
            return (d2*d2/(d1-d2))
        def solve():
            nonlocal tmp_1
            nonlocal res
            tmp_2 = x.copy()
            Move_Forward()
            while(r(tmp_1,tmp_2,x)>e):
                tmp_1 = tmp_2.copy()
                tmp_2 = x.copy()
                Move_Forward()
            res = r(tmp_1,tmp_2,x)
        solve()
    res = np.linalg.norm(np.array([[1] for i in range(n)]) - x, ord=2)
    return (x,count,res)

'''Test of Gauss Method'''
'''
A = Defaul_Matrix.Default_matrix(32)
b = np.dot(A,np.array([[1] for i in range(1024)],dtype='float64'))
x = Gauss_Method(A,b,1e-6,0)
print(x)


GaussOut = open('Gauss_0.csv','w',newline='')
Out = csv.writer(GaussOut,dialect='excel')
for i in range(3,21):
    A = Defaul_Matrix.Default_matrix(i)
    b = np.dot(A,np.array([[1] for i in range(i*i)],dtype='float64'))
    x = Gauss_Method(A,b,1e-6,0)
    res = [i,x[1],x[2]]
    Out.writerow(res)

GaussOut = open('Gauss_1.csv','w',newline='')
Out = csv.writer(GaussOut,dialect='excel')
for i in range(3,21):
    A = Defaul_Matrix.Default_matrix(i)
    b = np.dot(A,np.array([[1] for i in range(i*i)],dtype='float64'))
    x = Gauss_Method(A,b,1e-6,1)
    res = [i,x[1],x[2]]
    Out.writerow(res)

GaussOut = open('Gauss_2.csv', 'w', newline='')
Out = csv.writer(GaussOut, dialect='excel')
for i in range(3, 21):
    A = Defaul_Matrix.Default_matrix(i)
    b = np.dot(A, np.array([[1] for i in range(i * i)], dtype='float64'))
    x = Gauss_Method(A, b, 1e-6, 2)
    res = [i, x[1], x[2]]
    Out.writerow(res)

JacobiOut = open('Jacobi_0.csv', 'w', newline='')
Out = csv.writer(JacobiOut, dialect='excel')
for i in range(3, 21):
    A = Defaul_Matrix.Default_matrix(i)
    b = np.dot(A, np.array([[1] for i in range(i * i)], dtype='float64'))
    x = Jacobi_Method(A, b, 1e-6, 0)
    res = [i, x[1], x[2]]
    Out.writerow(res)

JacobiOut = open('Jacobi_1.csv', 'w', newline='')
Out = csv.writer(JacobiOut, dialect='excel')
for i in range(3, 21):
    A = Defaul_Matrix.Default_matrix(i)
    b = np.dot(A, np.array([[1] for i in range(i * i)], dtype='float64'))
    x = Jacobi_Method(A, b, 1e-6, 1)
    res = [i, x[1], x[2]]
    Out.writerow(res)

JacobiOut = open('Jacobi_2.csv', 'w', newline='')
Out = csv.writer(JacobiOut, dialect='excel')
for i in range(3, 21):
    A = Defaul_Matrix.Default_matrix(i)
    b = np.dot(A, np.array([[1] for i in range(i * i)], dtype='float64'))
    x = Jacobi_Method(A, b, 1e-6, 2)
    res = [i, x[1], x[2]]
    Out.writerow(res)



A = Defaul_Matrix.Default_matrix(10)
b = np.array([[1] for i in range(100)],dtype='float64')
x = Jacobi_Method(A,b,1e-6,0)
x = Gauss_Method(A,b,1e-6,0)

'''
JacobiOut = open('32-Jacobi.csv', 'w', newline='')
GaussOut = open('32-Gauss.csv','w',newline='')
Out = csv.writer(JacobiOut, dialect='excel')
anOut = csv.writer(GaussOut,dialect='excel')
for j in range(3,11):
    A = Defaul_Matrix.Default_matrix(j)
    b = np.dot(A, np.array([[1] for i in range(j * j)], dtype='float64'))
    x = Jacobi_Method(A,b,1e-6,0)[1]
    Out.writerow([j,x])
    anOut.writerow([j,Gauss_Method(A,b,1e-6,0)[1]])
