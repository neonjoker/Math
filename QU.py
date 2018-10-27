import numpy as np
sgn = lambda x: 1 if x > 0 else -1 if x < 0 else 0

def CGS(A):
    n = A.shape[1]
    Q = A.copy()
    R = np.zeros((n,n))
    R[0,0] = np.linalg.norm(Q[:,0],ord = 2)
    Q[:,0] = Q[:,0] / R[0,0]
    for j in range(1,n):
        for i in range(0,j):
            R[i,j] = np.dot(Q[:,i].T,A[:,j])
            Q[:, j] = Q[:, j] - R[i, j] * Q[:, i]
        R[j,j] = np.linalg.norm(Q[:,j])
        Q[:,j] = Q[:,j] / R[j,j]
    return(Q,R)

def MGS(A):
    n = A.shape[1]
    Q = A.copy()
    R = np.zeros((n,n))
    for i in range(n-1):
        R[i,i] = np.linalg.norm(Q[:,i],ord = 2)
        Q[:,i] = Q[:,i] / R[i,i]
        for j in range(i+1,n):
            R[i,j] = np.dot(Q[:,i].T,Q[:,j])
            Q[:,j] = Q[:,j] - R[i,j] * Q[:,i]
    R[n-1,n-1] = np.linalg.norm(Q[:,n-1],ord = 2)
    Q[:,n-1] = Q[:,n-1] / R[n-1,n-1]
    return (Q,R)

def Householder(A):
    (r, c) = np.shape(A)
    Q = np.identity(r)
    R = np.copy(A)
    s = min(r,c)
    D = np.zeros(s - 1)
    for cnt in range(s - 1):
        D[cnt] = np.linalg.norm(R[cnt:,cnt])
        R[cnt,cnt] = (R[cnt,cnt] - D[cnt])
        R[cnt:,cnt] = R[cnt:,cnt] / np.linalg.norm(R[cnt:,cnt])
        for i in range(cnt+1,c):
            R[cnt:,i] = R[cnt:,i] - 2 * np.dot(R[cnt:,cnt].T,R[cnt:,i]) * R[cnt:,cnt]

    for cnt in range(s - 2,-1,-1):
        for i in range(cnt,c):
            Q[cnt:, i] = Q[cnt:, i] - 2 * np.dot(R[cnt:, cnt].T, Q[cnt:, i]) * R[cnt:, cnt]
    for i in range(s-1):
        R[i,i] = D[i]
    R = np.triu(R)
    return (Q, R)

def Givens(A):
    (r, c) = np.shape(A)
    s = min(r,c)
    Q = np.identity(s)
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
                for i in range(col+1,c):
                    (a, b) = (R[col, i], R[row, i])
                    R[col, i] = cos * a - sin * b
                    R[row, i] = cos * b + sin * a
    for col in range(s-2,-1,-1):
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
                Q[row, i] = cos * b - sin * a
    R = np.triu(R)
    return (Q, R)

'''
from Defaul_Matrix import Default_matrix
A = Default_matrix(3)
res = Givens(A)
Q = res[0]
R = res[1]
print(R)
'''