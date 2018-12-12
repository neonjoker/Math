from sympy import *
import numpy as np

def Newton(x_0,var,expr,TOL = 1e-10,m = 10000,out = False):
    if(out != True):
        if(len(var) != len(expr)):
            raise ValueError('len(x) != len(expr), Please check your input value.')
        else:
            dif = [[diff(expr[i],var[j]) for j in range(len(var))] for i in range(len(expr))]
            dif = lambdify(var,dif)
            f = lambdify(var,expr)
            x = x_0
            for k in range(m):
                y = np.linalg.solve(np.array(dif(*x)), -np.array(f(*x)))
                x = x + y
                if(np.linalg.norm(y) < TOL):
                    return x
            raise ValueError('Maximum number of iterations exceed')
    else:
        if(len(var) != len(expr)):
            raise ValueError('len(x) != len(expr), Please check your input value.')
        else:
            dif = [[diff(expr[i],var[j]) for j in range(len(var))] for i in range(len(expr))]
            dif = lambdify(var,dif)
            f = lambdify(var,expr)
            x = x_0
            data = [x]
            for k in range(m):
                y = np.linalg.solve(np.array(dif(*x)), -np.array(f(*x)))
                x = x + y
                data.append(x)
                if(np.linalg.norm(y) < TOL):
                    return (x,data)
            raise ValueError('Maximum number of iterations exceed')

def Broyden(x_0,var,expr,TOL = 1e-10,m = 10000,out = False):
    if(out != True):
        if (len(var) != len(expr)):
            raise ValueError('len(var) != len(expr), Please check your input value.')
        else:
            dif = [[diff(expr[i], var[j]) for j in range(len(var))] for i in range(len(expr))]
            dif = lambdify(var, dif)
            f = lambdify(var, expr)
            A, v, x = np.array(dif(*x_0)), np.array(f(*x_0)), x_0
            H = np.linalg.inv(A)
            s = -np.dot(H, v)
            x = x + s
            n = len(var)
            for k in range(m):
                w = v
                v = np.array(f(*x))
                y = v - w
                z = -np.dot(H, y)
                p = -np.dot(s, z)
                if (p != 0):
                    C = p * np.identity(n) + np.outer(s + z, s)
                    H = (1 / p) * np.dot(C, H)
                    s = -np.dot(H, v)
                    x = x + s
                    if (np.linalg.norm(s) < TOL):
                        return x
                else:
                    raise ValueError('Method Failed')
            raise ValueError('Maximum number of iterations exceeded.')
    else:
        if (len(var) != len(expr)):
            raise ValueError('len(var) != len(expr), Please check your input value.')
        else:
            dif = [[diff(expr[i], var[j]) for j in range(len(var))] for i in range(len(expr))]
            dif = lambdify(var, dif)
            f = lambdify(var, expr)
            A, v, x = np.array(dif(*x_0)), np.array(f(*x_0)), x_0
            data = [x]
            H = np.linalg.inv(A)
            s = -np.dot(H, v)
            x = x + s
            data.append(x)
            n = len(var)
            for k in range(m):
                w = v
                v = np.array(f(*x))
                y = v - w
                z = -np.dot(H, y)
                p = -np.dot(s, z)
                if (p != 0):
                    C = p * np.identity(n) + np.outer(s + z, s)
                    H = (1 / p) * np.dot(C, H)
                    s = -np.dot(H, v)
                    x = x + s
                    data.append(x)
                    if (np.linalg.norm(s) < TOL):
                        return (x,data)
                else:
                    raise ValueError('Method Failed')
            raise ValueError('Maximum number of iterations exceeded.')


def Modified_Newton(x_0,var,expr,r = 2,TOL = 1e-10,m = 10000,out = False):
    if(out != True):
        if (len(var) != len(expr)):
            raise ValueError('len(x) != len(expr), Please check your input value.')
        else:
            dif = [[diff(expr[i], var[j]) for j in range(len(var))] for i in range(len(expr))]
            dif = lambdify(var, dif)
            f = lambdify(var, expr)
            x = x_0
            for k in range(m):
                B = np.linalg.inv(np.array(dif(*x)))
                x_1 = x
                for j in range(r):
                    y = np.dot(B, f(*x_1))
                    x_1 = x_1 - y
                    if(np.linalg.norm(y) < TOL):
                        return x_1
                x = x_1
            raise ValueError('Maximum number of iterations exceed')
    else:
        if (len(var) != len(expr)):
            raise ValueError('len(x) != len(expr), Please check your input value.')
        else:
            dif = [[diff(expr[i], var[j]) for j in range(len(var))] for i in range(len(expr))]
            dif = lambdify(var, dif)
            f = lambdify(var, expr)
            x = x_0
            data = [x]
            for k in range(m):
                B = np.linalg.inv(np.array(dif(*x)))
                x_1 = x
                for j in range(r):
                    y = np.dot(B, f(*x_1))
                    x_1 = x_1 - y
                    data.append(x_1)
                    if(np.linalg.norm(y) < TOL):
                        return (x_1,data)
                x = x_1
            raise ValueError('Maximum number of iterations exceed')


def Discrete_Newton(x_0,var,expr,h,TOL = 1e-10,m = 10000,out = False):
    if(out != True):
        if(len(var) != len(expr)):
            raise ValueError('len(x) != len(expr), Please check your input value.')
        else:
            I = np.identity(len(var))
            f = [lambdify(var,expr[i]) for i in range(len(expr))]
            x = x_0
            for k in range(m):
                fx = [f[i](*x) for i in range(len(var))]
                dif = [[((f[i](*(x + h[i] * I[j])) - fx[i]) / h[i]) for j in range(len(var))] for i in range(len(expr))]
                y = np.linalg.solve(np.array(dif), -np.array(fx))
                x = x + y
                if(np.linalg.norm(y) < TOL):
                    return x
            raise ValueError('Maximum number of iterations exceed')
    else:
        if (len(var) != len(expr)):
            raise ValueError('len(x) != len(expr), Please check your input value.')
        else:
            I = np.identity(len(var))
            f = [lambdify(var, expr[i]) for i in range(len(expr))]
            x = x_0
            data = [x]
            for k in range(m):
                fx = [f[i](*x) for i in range(len(var))]
                dif = [[((f[i](*(x + h[i] * I[j])) - fx[i]) / h[i]) for j in range(len(var))] for i in range(len(expr))]
                y = np.linalg.solve(np.array(dif), -np.array(fx))
                x = x + y
                data.append(x)
                if (np.linalg.norm(y) < TOL):
                    return (x, data)
            raise ValueError('Maximum number of iterations exceed')


def Secant_2p(x_0,var,expr,h,TOL = 1e-10,m = 10000,out = False):
    if(out != True):
        if(len(var) != len(expr)):
            raise ValueError('len(x) != len(expr), Please check your input value.')
        else:
            I = np.identity(len(var))
            f = [lambdify(var,expr[i]) for i in range(len(expr))]
            x = x_0
            fx = np.array([f[i](*x) for i in range(len(var))])
            dif = [[((f[i](*(x + h[i] * I[j])) - fx[i]) / h[i]) for j in range(len(var))] for i in range(len(expr))]
            y = np.linalg.solve(np.array(dif), -fx)
            x = x + y
            for k in range(m):
                fx = np.array([f[i](*x) for i in range(len(var))])
                dif = [[((f[i](*(x + y[j] * I[j])) - fx[i]) / y[j]) for j in range(len(var))] for i in range(len(expr))]
                y = np.linalg.solve(np.array(dif), -fx)
                x = x + y
                if(np.linalg.norm(y) < TOL):
                    return x
            raise ValueError('Maximum number of iterations exceed')
    else:
        if (len(var) != len(expr)):
            raise ValueError('len(x) != len(expr), Please check your input value.')
        else:
            I = np.identity(len(var))
            f = [lambdify(var, expr[i]) for i in range(len(expr))]
            x = x_0
            data = [x]
            fx = np.array([f[i](*x) for i in range(len(var))])
            dif = [[((f[i](*(x + h[i] * I[j])) - fx[i]) / h[i]) for j in range(len(var))] for i in range(len(expr))]
            y = np.linalg.solve(np.array(dif), -fx)
            x = x + y
            data.append(x)
            for k in range(m):
                fx = np.array([f[i](*x) for i in range(len(var))])
                dif = [[((f[i](*(x + y[j] * I[j])) - fx[i]) / y[j]) for j in range(len(var))] for i in range(len(expr))]
                y = np.linalg.solve(np.array(dif), -fx)
                x = x + y
                data.append(x)
                if (np.linalg.norm(y) < TOL):
                    return (x, data)
            raise ValueError('Maximum number of iterations exceed')


if __name__ == '__main__':
    '''
    x_1, x_2 ,x_3= symbols('x_1 x_2 x_3')
    expr1 = 3*x_1 - cos(x_2*x_3) - Integer(1)/2
    expr2 = x_1**2 - 81*(x_2+0.1)**2 + sin(x_3) + 1.06
    expr3 = E**(-x_1 * x_2) + 20 * x_3 + (10 * pi - Integer(3))/ 3
    expr = (expr1, expr2, expr3)
    var = (x_1,x_2,x_3)
    x_0 = np.array([0.1,0.1,-0.1])
    print(Secant_2p(x_0, var, expr,1e-3 * np.ones(len(var)),out = True))
    '''
    from Defaul_Matrix import T_matrix

    def T_Eigfunc(l, *x):
        n = len(x[0])
        T = T_matrix(n)
        x_ = x[0]
        res = np.dot((T - l * np.identity(n)), x_)
        return (np.append(res, np.dot(x_, x_) - 1))


    def DifT_Eigfunc(l, *x):
        n = len(x[0])
        T = T_matrix(n) - l * np.identity(n)
        return (np.column_stack((np.row_stack((T, 2 * x[0])), np.append(-1 * x[0], 0))))


    x = np.array([1, 0, 0, 0, 0])
    l = np.dot(np.dot(np.transpose(x), T_matrix(5)), x)
    np.array(DifT_Eigfunc(l, x))

    for k in range(10000):
        y = np.linalg.solve(np.array(DifT_Eigfunc(l, x)), -np.array(T_Eigfunc(l, x)))
        x = x + y[:-1]
        l = l + y[-1]
        if (np.linalg.norm(y) < 1e-6):
            break
    print(x, l)