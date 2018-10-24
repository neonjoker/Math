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
            R[i,j] = np.dot(np.transpose(Q[:,i]),A[:,j])
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
            R[i,j] = np.dot(np.transpose(Q[:,i]),Q[:,j])
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
    n = A.shape[1]
    m = A.shape[0]
    s = min(n,m) - 1
    d = np.zeros(s)
    b = np.zeros(s)
    def householder(a):
        alpha = -sgn(a[0]) * np.linalg.norm(a,ord = 2)
        b = alpha * alpha - alpha * a[0]
        a[0] = a[0] - alpha
        return (alpha,b)
    def multiply_householder(a,b,u):
        a =  a - np.dot(np.transpose(u),a) / b * u
    for i in range(s):
        res = householder(A[i:m:,i])
        d[i] = res[0]
        b[i] = res[1]
        for j in range(i+1,n):
            A[i:m:,j] = A[i:m:,j] - np.dot(np.transpose(A[i:m:,i]),A[i:m:,j])/ b[i] * A[i:m,i]
    R = np.triu(A)
    for i in range(s):
        R[i,i] = d[i]
    Q = np.ones((m,m),dtype='float64')
    for i in range(s):
        for j in range(i,m):
            multiply_householder(Q[i:m:,j],b[i],A[i:m:,i])
    return (Q,R)
