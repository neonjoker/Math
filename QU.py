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
    for i in range(n):
        R[i,i] = np.linalg.norm(Q[:,i],ord = 2)
        Q[:,i] = Q[:,i] / R[i,i]
        for j in range(i+1,n):
            R[i,j] = np.dot(Q[:,i].T,Q[:,j])
            Q[:,j] = Q[:,j] - R[i,j] * Q[:,i]
    R[n-1,n-1] = np.linalg.norm(Q[:,n-1],ord = 2)
    Q[:,n-1] = Q[:,n-1] / R[n-1,n-1]
    return (Q,R)

def Householder(A):
    n = A.shape[1]
    m = A.shape[0]
    d = np.zeros(n)
    Q = A.copy()
    for k in range(n-1):
        delta = - sgn(Q[k,k]) * np.linalg.norm(Q[k:,k],ord = 2)
        h = delta - Q[k,k]
        Q[k,k] = -h
        d[k] = delta

def Householder(A):
    (m,n) = A.shape
    s = min(n,m) - 1
    d = np.zeros(s)
    b = np.zeros(s)

    def householder(a):
        alpha = -sgn(a[0]) * np.linalg.norm(a,ord = 2)
        b = alpha * alpha - alpha * a[0]
        a[0] = a[0] - alpha
        return (alpha,b)

    def multiply_householder(a,b,u):
        return  a - np.dot(np.transpose(u),a) / b * u

    for i in range(s):
        res = householder(A[i:m:,i])
        d[i] = res[0]
        b[i] = res[1]
        for j in range(i+1,n):
            A[i:m:,j] = A[i:m:,j] - np.dot(A[i:m:,i].T,A[i:m:,j])/ b[i] * A[i:m,i]
    R = np.triu(A)
    for i in range(s):
        R[i,i] = d[i]
    Q = np.identity(m)
    for i in range(s):
        for j in range(i,m):
            Q[i:m:,j] = multiply_householder(Q[i:m:,j],b[i],A[i:m:,i])
    return (Q,R)

def Givens(A):
    (r, c) = np.shape(A)
    Q = np.identity(r)
    R = np.copy(A)
    (rows, cols) = np.tril_indices(r, -1, c)
    for (row, col) in zip(rows, cols):
        if R[row, col] != 0:  # R[row, col]=0则c=1,s=0,R、Q不变
            r_ = np.hypot(R[col, col], R[row, col])  # d
            cos = R[col, col]/r_
            sin = -R[row, col]/r_
            for i in range(c):
                (a,b) = (R[col,i],R[row,i])

                R[col,i] = cos * a - sin * b
                R[row,i] = cos * b + sin * a
            (a, b) = (Q[:, col], Q[:, row])
            Q[:,col] = cos * a - sin * b
            Q[:,row] = cos * b + sin * a

    return (Q, R)
'''
from Defaul_Matrix import Default_matrix
A = Default_matrix(3)
res = givens_rotation(A)
Q = res[0]
R = res[1]
print(R)
'''