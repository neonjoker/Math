from Defaul_Matrix import T_matrix
import numpy as np
import QU
sgn = lambda x: 1 if x > 0 else -1 if x < 0 else 0

def Generate_Question(n):
    G = T_matrix(n,n-1)
    b = np.zeros(n)
    for k in range(n-1):
        b[k] = (k+1)/n
    b[0] = b[0] + 1
    b[n-2] = b[n-2] + 1
    b[n-1] = 0
    times = np.random.randint(10,21)
    for k in range(times):
        sin = np.random.rand()
        cos = np.sqrt(1 - sin * sin)
        i = np.random.randint(0, n)
        j = np.random.randint(0, n)
        while (j == i):
            j = np.random.randint(0, n)
        (tmp_1,tmp_2) = (b[j],b[i])
        b[j] = cos * tmp_1 + sin * tmp_2
        b[i] = cos * tmp_2 - sin * tmp_1
        for c in range(n-1):
            (tmp_1, tmp_2) = (G[j, c], G[i, c])
            G[j, c] = cos * tmp_1 + sin * tmp_2
            G[i, c] = cos * tmp_2 - sin * tmp_1
    return (G,b)

def solve_L(L,b):
    n = L.shape[0]
    x = np.zeros(n)
    x[0] = b[0] / L[0,0]
    for i in range(1,n):
        x[i] = b[i]
        for j in range(i):
            x[i] = x[i] - L[i,j] * x[j]
        x[i] = x[i] / L[i,i]
    return x

def solve_U(U,b):
    n = U.shape[0]
    x = np.zeros(n)
    x[n-1] = b[n-1] / U[n-1,n-1]
    for i in range(n-2,-1,-1):
        x[i] = b[i]
        for j in range(i+1,n):
            x[i] = x[i] - U[i,j] * x[j]
        x[i] = x[i] / U[i,i]
    return x

def Householder_solve(A,b):
    (r, c) = np.shape(A)
    qb = b.copy()
    #Q = np.identity(r)
    R = np.copy(A)
    s = min(r,c)
    D = np.zeros(s - 1)
    for cnt in range(s - 1):
        D[cnt] = np.linalg.norm(R[cnt:,cnt])
        R[cnt,cnt] = (R[cnt,cnt] - D[cnt])
        R[cnt:,cnt] = R[cnt:,cnt] / np.linalg.norm(R[cnt:,cnt])
        qb[cnt:] = qb[cnt:] - 2 * np.dot(R[cnt:,cnt].T,qb[cnt:]) * R[cnt:,cnt]
        for i in range(cnt+1,c):
            R[cnt:,i] = R[cnt:,i] - 2 * np.dot(R[cnt:,cnt].T,R[cnt:,i]) * R[cnt:,cnt]

    '''
    for cnt in range(s - 2,-1,-1):
        for i in range(cnt,c):
            Q[cnt:, i] = Q[cnt:, i] - 2 * np.dot(R[cnt:, cnt].T, Q[cnt:, i]) * R[cnt:, cnt]
    '''
    for i in range(s-1):
        R[i,i] = D[i]
    R = np.triu(R)
    x = solve_U(R[:s, :s], qb)
    return x

def Givens_solve(A,b):
    (r, c) = np.shape(A)
    s = min(r,c)
    qb = b.copy()
    #Q = np.identity(r)
    R = np.copy(A)
    for col in range(s - 1):
        for row in range(col+1 ,r):
            if (R[row, col] != 0):
                r_ = np.hypot(R[col, col], R[row, col])  # d
                cos = R[col, col] / r_
                sin = -R[row, col] / r_
                rho = 1
                if(cos != 0):
                    if(abs(sin)<abs(cos)):
                        rho = sgn(cos) * sin / 2
                    else:
                        rho = 2 * sgn(sin) / cos
                (R[col,col],R[row,col]) = (r_,rho)
                (tmp_1,tmp_2) = (qb[col],qb[row])
                qb[col] = cos * tmp_1 - sin * tmp_2
                qb[row] = cos * tmp_2 + sin * tmp_1
                for i in range(col+1,c):
                    (a, b) = (R[col, i], R[row, i])
                    R[col, i] = cos * a - sin * b
                    R[row, i] = cos * b + sin * a
    '''for col in range(s-2,-1,-1):
        for row in range(r-1,col,-1):
            (cos,sin) = (0,1)
            if(abs(R[row,col]) < 1):
                sin = 2 * R[row,col]
                cos = np.sqrt(1 - sin * sin)
            else:
                cos = 2 / R[row,col]
                sin = np.sqrt(1 - cos * cos)
            for i in range(col,c):
                (a, b) = (Q[col, i], Q[row, i])
                Q[col, i] = cos * a + sin * b
                Q[row, i] = cos * b - sin * a'''
    R = np.triu(R)
    x = solve_U(R[:s,:s],qb)
    return x


Question = Generate_Question(10)
A = Question[0]
b = Question[1]
'''
x = np.linalg.solve(np.dot(A.T,A),np.dot(A.T,b))
print(x)'''

