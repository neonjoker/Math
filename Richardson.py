import numpy as np


def Richardson(A,b,x,m):
    n = b.size()
    EigMax = 4 * (1-np.cos((np.sqrt(n)*np.pi)/(np.sqrt(n)+1)))
    EigMin = 4 * (1 - np.cos((np.pi) / (np.sqrt(n) + 1)))
    G = np.diag(np.diag(np.ones_like(A))) - (2 / (EigMax + EigMin)) * A
    g = (2 / (EigMin + EigMax)) * b
    ksi = (2 - EigMax - EigMin)/(EigMax - EigMin)
    rho = 2.0
    v = 2 / (2 - EigMax - EigMin)
    tmp_x = 2 / (EigMax + EigMin) * (b - A * x)
    for i in range(m):
        rho = 1 / (1 - (rho / (4 * ksi * ksi)))
        tmp = (1 - rho) * x + rho * (v * (np.dot(G,tmp_x) + g) + (1 - v) * tmp_x)
        x = np.copy(tmp_x)
        tmp_x = np.copy(tmp)
