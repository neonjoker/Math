import numpy as np
import Defaul_Matrix

def Jacobi_Method(A,b,e,Stop_Method):
    n = b.size
    x = np.array([[0] for i in range(n)],dtype='float64')
    def Move_Forward():
        nonlocal x
        tmp_x = b.copy()
        for i in range(n):
            for j in range(n):
                if (j != i):
                    tmp_x[i][0] = tmp_x[i][0] - A[i, j] * x[j][0]
            tmp_x[i][0] = tmp_x[i][0] / A[i, i]
        x = tmp_x.copy()
    tmp_1 = x.copy()
    Move_Forward()
    if(Stop_Method != 2):
        if(Stop_Method == 0):
            def r():
                rest = np.linalg.norm((np.dot(A,x) - b),ord = 2)
                return rest
            def solve():
                while(abs(r())>e):
                    Move_Forward()
            solve()
        elif(Stop_Method == 1):
            def r(x1,x2):
                return np.linalg.norm((x1-x2),ord = 2)
            def solve():
                nonlocal tmp_1
                while(r(x,tmp_1)>e):
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
            while(r(x,tmp_1,tmp_2)<e):
                Move_Forward()
                tmp_2 = tmp_1.copy()
                tmp_1 = x.copy()
        solve()
    return x

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
    def Move_Forward():
        nonlocal x
        for i in range(n):
            x[i][0] = b[i][0]
            for j in range(n):
                if (j != i):
                    x[i][0] = x[i][0] - A[i, j] * x[j][0]
            x[i][0] = x[i][0] / A[i, i]
        x = x.copy()
    tmp_1 = x.copy()
    Move_Forward()
    if(Stop_Method != 2):
        if(Stop_Method == 0):
            def r():
                rest = np.linalg.norm((np.dot(A,x) - b),ord = 2)
                return rest
            def solve():
                while(abs(r())>e):
                    Move_Forward()
            solve()
        elif(Stop_Method == 1):
            def r(x1,x2):
                return np.linalg.norm((x1-x2),ord = 2)
            def solve():
                nonlocal tmp_1
                while(r(x,tmp_1)>e):
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
            while(r(x,tmp_1,tmp_2)<e):
                Move_Forward()
                tmp_2 = tmp_1.copy()
                tmp_1 = x.copy()
        solve()
    return x

'''Test of Gauss Method'''

A = Defaul_Matrix.Default_matrix(32)
b = np.dot(A,np.array([[1] for i in range(1024)],dtype='float64'))
x = Gauss_Method(A,b,1e-6,2)
print(x)

