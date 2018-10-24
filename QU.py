import numpy as np

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

