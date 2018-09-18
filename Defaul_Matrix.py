import numpy as np
def T_matrix(n):
    t = np.zeros((n,n))
    for i in range(n-1):
        t[i,i] = 2
        t[i,i+1] = -1
        t[i+1,i] = -1
    t[n-1,n-1] = 2
    return t

def I_matrix(n):
    I = np.zeros((n,n))
    for i in range(n):
        I[i,i] = 1
    return I

def Default_matrix(n):
    T = T_matrix(n)
    I = I_matrix(n)
    return (np.kron(T,I) + np.kron(I,T))
