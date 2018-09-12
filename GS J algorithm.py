import numpy as np
import Defaul_Matrix

def Jacobi_Method(A,b,Stop_Method):
    n = b.ndim
    x = np.array([0 for i in range(n)])
    if(Stop_Method == 0):
