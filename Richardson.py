import numpy as np

def Richardson(A,b,x,m):
    n = b.size()
    EigMax = 4 * (1-np.cos((np.sqrt(n)*np.pi)/(np.sqrt(n)+1)))
    EigMin = 4 * (1 - np.cos((np.pi) / (np.sqrt(n) + 1)))
    ksi = (2 - EigMax - EigMin)/(EigMax - EigMin)
    rho = 2.0
    v = 2 / (2 - EigMax - EigMin)
    tmp_x = 2 / (EigMax + EigMin) * (b - A * x)
    for i in range(m):
        rho = 1 / (1 - (rho / (4 * ksi * ksi)))
        x = (1 - rho) * x + rho * (v * )
