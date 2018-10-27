import numpy as np

def T_matrix(*shape):
    if(len(shape)<1):
        raise ValueError('dim < 1')
    l = len(shape)
    if(len(shape)==1):
        n = shape[0]
        t = np.zeros((n,n))
        for i in range(n-1):
            t[i,i] = 2
            t[i,i+1] = -1
            t[i+1,i] = -1
        t[n-1,n-1] = 2
    else:
        n = shape[0]
        m = shape[1]
        s = min(n,m)
        t = np.zeros((n,m))
        for i in range(s-1):
            t[i, i] = 2
            t[i, i + 1] = -1
            t[i + 1, i] = -1
        t[s-1,s-1] = 2
        if(n>m):
            t[s,m-1] = -1
        else:
            t[n-1,s] = -1
    return t

def I_matrix(n):
    I = np.zeros((n,n))
    for i in range(n):
        I[i,i] = 1
    return I

def Default_matrix(n):
    T = T_matrix(n)
    I = np.identity(n)
    return (np.kron(T,I) + np.kron(I,T))
