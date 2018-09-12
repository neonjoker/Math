import numpy as np
import Defaul_Matrix

def Jacobi_Method(A,b,Stop_Method):
    n = b.ndim
    x = np.array([0 for i in range(n)])
    if(Stop_Method == 0):
        def r(x):
            return np.linalg.norm((np.dot(A,x) - b),ord=2)
    elif(Stop_Method == 1):
        def r(x1,x2):
            return np.linalg.norm((x1-x2),ord=2)
    elif(Stop_Method == 2):
        def r(x1,x2):
            return np.linalg.norm((np.dot(A,x)))
