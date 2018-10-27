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

def Givens_Random(n):
    sin = np.random.rand()
    cos = np.sqrt(1 - sin * sin)
    G = np.identity(n)
    i = np.random.randint(0,n)
    j = np.random.randint(0,n)
    while(j == i):
        j = np.random.randint(0,n)
    (G[i,i],G[j,j]) = (cos,cos)
    (G[i,j],G[j,i]) = (sin, -sin)
    return G

def Givens_RandomMultiply(dim,n=1):
    G = np.identity(dim)
    for k in range(n):
        sin = np.random.rand()
        cos = np.sqrt(1 - sin * sin)
        i = np.random.randint(0, dim)
        j = np.random.randint(0, dim)
        while (j == i):
            j = np.random.randint(0, dim)
        for c in range(dim):
            (a, b) = (G[j, c], G[i, c])
            G[j, c] = cos * a + sin * b
            G[i, c] = cos * b - sin * a
    return G

G = Givens_Random(2)