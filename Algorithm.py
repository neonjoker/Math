import numpy as np

def iterator(A,b,x,e,Move_Forward,Stop_Method=0):
    x = x.copy()
    tmp_1 = x.copy()
    Move_Forward(x)
    if (Stop_Method != 2):
        if (Stop_Method == 0):
            def r():
                rest = np.linalg.norm((np.dot(A, x) - b), ord=2)
                return rest

            def solve():
                while (abs(r()) > e):
                    Move_Forward(x)

            solve()
        elif (Stop_Method == 1):
            def r(x1, x2):
                return np.linalg.norm((x1 - x2), ord=2)

            def solve():
                nonlocal tmp_1
                while (r(x, tmp_1) > e):
                    tmp_1 = x.copy()
                    Move_Forward(x)

            solve()
    elif (Stop_Method == 2):
        def r(x1, x2, x3):
            d1 = np.linalg.norm((x1 - x2), ord=2)
            d2 = np.linalg.norm((x2 - x3), ord=2)
            return (d2 * d2 / (d1 - d2))

        def solve():
            Move_Forward(x)
            nonlocal tmp_1
            tmp_2 = tmp_1.copy()
            tmp_1 = x.copy()
            while (r(x, tmp_1, tmp_2) < e):
                Move_Forward(x)
                tmp_2 = tmp_1.copy()
                tmp_1 = x.copy()

        solve()
    return x

def Gauss_Method(A,b,e=1e-6,Stop_Method=0):
    n = b.size
    x = np.zeros((n,1),dtype='float64')
    def Move_Forward(x):
        for i in range(n):
            x[i][0] = b[i][0]
            for j in range(n):
                if (j != i):
                    x[i][0] = x[i][0] - A[i, j] * x[j][0]
            x[i][0] = x[i][0] / A[i, i]
        x = x.copy()
    res = iterator(A,b,x,e,Move_Forward,Stop_Method)
    return res

def CG(A,b,e=1e-6,Stop_Method=0):
    n = b.size
    x = np.zeros(n,dtype='float64')
    r = np.dot(A, b) - b
    p = np.copy(r)
    def Move_Forward(x):
        nonlocal r,p
        q = np.dot(A,p)
        a = -np.dot(np.transpose(r),p)/np.dot(np.transpose(p),q)[0][0]
        for i in range(n):
            x[i] = x[i] + a * p[i][0]
        for i in range(n):
            r[i][0] + a * q[i][0]
        beta = np.dot(np.transpose(r),q)[0][0]/np.dot(np.transpose(p),q)[0][0]
        for i in range(n):
            p[i][0] = -r[i][0] + beta * p[i][0]
    res = iterator(A, b, x, e, Move_Forward, Stop_Method)
    return res


from Defaul_Matrix import Default_matrix
A = Default_matrix(3)
b = np.dot(A,np.array([[1] for i in range(9)]))
res = CG(A,b)
print(res)