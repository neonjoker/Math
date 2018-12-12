import numpy as np
from scipy.special import comb
from sympy import *

def Euler(x_0,var,expr,N,start,end):
    a, b = start, end
    h = (b - a) / N
    t ,y = a, x_0
    f = lambdify(var, expr)
    res = []
    res.append(np.array([t, y]))
    for i in range(1, N + 1):
        y = y + h * f(t, y)
        t = a + i * h
        res.append(np.array([t, y]))
    return res

def Trapezoid(x_0,var,expr,N,start,end):
    a, b = start, end
    h = (b - a) / N
    t ,y = a, x_0
    f = lambdify(var, expr)
    res = [np.array([t, y])]
    for i in range(1, N + 1):
        y_0 = y + h * f(t, y)
        t = a + i * h
        y_1 = y + h * f(t, y_0)
        y = (y_0 + y_1) / 2
        res.append(np.array([t, y]))
    return res

def RK3(x_0,var,expr,N,start,end):
    a, b = start, end
    h = (b - a) / N
    t, y = a, x_0
    f = lambdify(var, expr)
    res = [np.array([t, y])]
    for i in range(1, N + 1):
        K_1 = f(t, y)
        K_2 = f(t + h/2, y + h/2 * K_1)
        K_3 = f(t + 3 * h/4, y + 3 * h/4 * K_2)
        y = y + (2 * K_1 + 3 * K_2 + 4 * K_3) * h / 9
        t = a + i * h
        res.append(np.array([t,y]))
    return res

def RK4(x_0,var,expr,N,start,end):
    a, b = start, end
    h = (b - a) / N
    t, y = a, x_0
    f = lambdify(var, expr)
    res = [np.array([t, y])]
    for i in range(1, N + 1):
        K_1 = f(t, y)
        K_2 = f(t + h/2, y + h/2 * K_1)
        K_3 = f(t + h/2, y + h/2 * K_2)
        K_4 = f(t + h, y + h * K_3)
        y = y + (K_1 + 2 * K_2 + 2 * K_3 + K_4) * h / 6
        t = a + i * h
        res.append(np.array([t,y]))
    return res

def Adams_3_3(x_0,var,expr,N,start,end):
    a, b = start, end
    h = (b - a) / N
    t, y = a, x_0
    f = lambdify(var, expr)
    res = [np.array([t, y])]
    for i in range(1, 3):
        K_1 = f(t, y)
        K_2 = f(t + h / 2, y + h / 2 * K_1)
        K_3 = f(t + 3 * h / 4, y + 3 * h / 4 * K_2)
        y = y + (2 * K_1 + 3 * K_2 + 4 * K_3) * h / 9
        t = a + i * h
        res.append(np.array([t, y]))
    for i in range(3, N+1):
        y = y + (23 * f(*(res[-1])) - 16 * f(*res[-2]) + 5 * f(*res[-3])) * h / 12
        t = a + i * h
        res.append(np.array([t, y]))
    return res

if __name__ == '__main__':
    u, t = symbols('u t')
    expr = 1 - 2 * t * u / (1 + t ** 2)
    var = (t, u)
    print(Adams_3_3(0, var, expr ,20, 0, 5))