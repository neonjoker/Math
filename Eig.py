import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt
from QU import MGS
sgn = lambda x: 1 if x > 0 else -1 if x < 0 else 0

def EigOfTmatrix(T):
    n = T.shape[1]
    l = np.zeros(n)
    for i in range(n):
        l[i] = 2 * (1 - np.cos((i+1) * np.pi / (n+1)))
    return l

def Power_Method(A,u,e,m,*out):
    #out参数第一个为主特征值
    if(out):
        eigvals = np.array([],dtype='float64')
        dist = np.array([])
        for k in range(m):
            b = max(abs(u))
            u = u / b
            v = np.dot(A, u)
            tmp_u = u.copy()
            b = max(abs(v))
            eigvals = np.append(eigvals,out[0]-b)
            if (b == 0):
                raise ValueError('A has eigenvalue 0,select new initial vector and restart')
            w = v / b
            dist = np.append(dist,np.sqrt(1 - ((np.dot(tmp_u , w))/( np.linalg.norm(w) * np.linalg.norm(tmp_u)))**2))
            if (abs(out[0]-b) < e):
                data = {'Eigvalue':eigvals,'IterateTimes':np.arange(1,k+2),'Distance':dist}
                df = pd.DataFrame(data)
                return (b, w, df)
            else:
                u = w
        data = {'Eigvalue': eigvals, 'IterateTimes': np.arange(1, m + 1),'Distance':dist}
        df = pd.DataFrame(data)
        print('failed!')
        return (b, w, df)
    else:
        for k in range(m):
            b = max(abs(u))
            u = u / b
            v = np.dot(A, u)
            b = max(abs(v))
            if (b == 0):
                raise ValueError('A has eigenvalue 0,select new initial vector and restart')
            w = v / b
            if (np.max(u - w) < e):
                return (b, w)
            else:
                u = w
        raise ValueError('Maximum number of iteration exceeded')


def Power_Method_Aitken(A,u,e,m,*out):
    if(out):
        eigvals = np.array([], dtype='float64')
        dist = np.array([])
        b = max(abs(u))
        b_0 = 0.0
        b_1 = 0.0
        u = u / b
        for k in range(m):
            tmp_u = u.copy()
            v = np.dot(A, u)
            b = max(abs(v))
            b_ = b_0 - (b_1 - b_0) * (b_1 - b_0) / (b - 2 * b_1 + b_0)
            eigvals = np.append(eigvals, out[0] - b_)
            if (b == 0):
                raise ValueError('A has eigenvalue 0,select new initial vector and restart')
            if (k == 0):
                w = v / b
            else:
                w = v / b
            dist = np.append(dist,np.sqrt(1 - ((np.dot(tmp_u, w)) / (np.linalg.norm(w) * np.linalg.norm(tmp_u))) ** 2))
            if (np.linalg.norm(abs(out[0]-b) < e)):
                data = {'Eigvalue': eigvals, 'IterateTimes': np.arange(1, k + 2),'Distance':dist}
                df = pd.DataFrame(data)
                return (b_, w , df)
            else:
                u = w
                b_0 = b_1
                b_1 = b
        data = {'Eigvalue': eigvals, 'IterateTimes': np.arange(1, m + 1),'Distance':dist}
        df = pd.DataFrame(data)
        print('failed!')
        return (b_, w, df)
    else:
        b = max(abs(u))
        b_0 = 0.0
        b_1 = 0.0
        u = u / b
        for k in range(m):
            v = np.dot(A, u)
            b = max(abs(v))
            b_ = b_0 - (b_1 - b_0) * (b_1 - b_0) / (b - 2 * b_1 + b_0)
            if (b == 0):
                raise ValueError('A has eigenvalue 0,select new initial vector and restart')
            if (k == 0):
                w = v / b
            else:
                w = v / b
            if (np.linalg.norm(u - w) < e):
                return (b_, w)
            else:
                u = w
                b_0 = b_1
                b_1 = b
        print('failed!')


def Power_Method_Rayleigh(A,u,e,m,*out):
    if(out):
        eigvals = np.array([], dtype='float64')
        dist = np.array([])
        for k in range(m):
            tmp_u = u.copy()
            r = np.dot(np.dot(np.transpose(u), A), u) / np.linalg.norm(u, ord=2)
            u = u / r
            v = np.dot(A, u)
            r = np.dot(np.dot(np.transpose(v), A), v) / np.linalg.norm(v, ord=2)
            eigvals = np.append(eigvals, out[0] - r)
            if (r == 0):
                raise ValueError('A has eigenvalue 0,select new initial vector and restart')
            w = v / r
            dist = np.append(dist, np.sqrt(1 - ((np.dot(tmp_u, w)) / (np.linalg.norm(w) * np.linalg.norm(tmp_u))) ** 2))
            if ((abs(out[0]-r)) < e):
                data = {'Eigvalue': eigvals, 'IterateTimes': np.arange(1, k + 2),'Distance':dist}
                df = pd.DataFrame(data)
                return (r, w, df)
            else:
                u = w
        data = {'Eigvalue': eigvals, 'IterateTimes': np.arange(1, m + 1),'Distance':dist}
        df = pd.DataFrame(data)
        print('failed!')
        return (r, w, df)
    else:
        for k in range(m):
            r = np.dot(np.dot(np.transpose(u), A), u) / np.linalg.norm(u, ord=2)
            u = u / r
            v = np.dot(A, u)
            r = np.dot(np.dot(np.transpose(v), A), v) / np.linalg.norm(v, ord=2)
            if (r == 0):
                raise ValueError('A has eigenvalue 0,select new initial vector and restart')
            w = v / r
            if (np.max(abs(u - w)) < e):
                return (r, w)
            else:
                u = w
        raise ValueError('Maximum number of iteration exceeded')

def Wieldant(A,u,e,m,*out):
    if(out):
        (l, x) = Power_Method(A, u, e, m)
        t = np.linalg.norm(x)
        u = x.copy()
        u[0] = u[0] - t
        b = 1 / (t * (t - x[0]))
        S = np.identity(A.shape[0]) - b * np.outer(u, u)
        B = np.dot(np.dot(S, A), S.T)
        return Power_Method(B[1:, 1:], u[1:], e, m ,out[0])
    else:
        (l, x) = Power_Method(A, u, e, m)
        t = np.linalg.norm(x)
        u = x.copy()
        u[0] = u[0] - t
        b = 1 / (t * (t - x[0]))
        S = np.identity(A.shape[0]) - b * np.outer(u, u)
        B = np.dot(np.dot(S, A), S.T)
        return Power_Method(B[1:, 1:], u[1:], e, m)

def inv_Power_Method(A,u,q,e,m,*out):
    if(out):
        eigvals = np.array([], dtype='float64')
        for k in range(m):
            v = np.linalg.solve((A - q * np.identity(A.shape[0])), u)
            a = max(abs(v))
            w = v / a
            eigvals = np.append(eigvals, out[0] - 1 / a - q)
            if (abs(1/a + q - out[0]) < e):
                data = {'Eigvalue': eigvals, 'IterateTimes': np.arange(1, k + 2)}
                df = pd.DataFrame(data)
                return (1 / a + q, w, df)
            else:
                u = w
        print('failed!')
        data = {'Eigvalue': eigvals, 'IterateTimes': np.arange(1, m + 1)}
        df = pd.DataFrame(data)
        return (1 / a + q, w,df)
    else:
        for k in range(m):
            v = np.linalg.solve((A - q * np.identity(A.shape[0])), u)
            a = max(abs(v))
            w = v / a
            if ((np.max(u) - np.max(w)) < e):
                return (1 / a + q, w)
            else:
                u = w
        print('failed!')
        return (1 / a + q, w)


def simultaneous_iteration(A,e,*out):
    if(out):
        n = A.shape[1]
        u_1 = np.random.rand(n)
        u_2 = np.random.rand(n)
        u_1 = u_1 / np.linalg.norm(u_1)
        tmp = u_2 - np.dot(u_1, u_2) * u_1
        u_2 = tmp / np.linalg.norm(tmp)
        v = np.vstack((u_1, u_2)).T
        l_1 = 0
        l_2 = 0
        tmp_l_1 = e + 1
        tmp_l_2 = e + 1
        data_1 = np.array([])
        data_2 = np.array([])
        count = 0
        while ((abs(tmp_l_1 - l_1) > e) or (abs(tmp_l_2 - l_2) > e)):
            l_1 = tmp_l_1
            l_2 = tmp_l_2
            u = np.dot(A, v)
            B = np.dot(v.T, u)
            ksi = (B[0, 0] - B[1, 1]) / (2 * B[0, 1])
            t = 1 / (2 * ksi)
            c = 1 / np.sqrt(1 + t * t)
            s = c * t
            tmp_l_1 = B[0, 0] * c * c + 2 * B[0, 1] * c * s + B[1, 1] * s * s
            tmp_l_2 = B[0, 0] * s * s - 2 * B[0, 1] * c * s + B[1, 1] * c * c
            data_1 = np.append(data_1,abs(l_1 - out[0]))
            data_2 = np.append(data_2,abs(l_2 - out[1]))
            u = np.dot(u, np.array([[c, s], [-s, c]]))
            v = MGS(u)[0]
            count = count + 1
        data = {'iteTimes':np.arange(1,count+1),'lamda_1':data_1,'lamda_2':data_2}
        data = pd.DataFrame(data)
        return ((l_1, l_2,data))
    else:
        n = A.shape[1]
        u_1 = np.random.rand(n)
        u_2 = np.random.rand(n)
        u_1 = u_1 / np.linalg.norm(u_1)
        tmp = u_2 - np.dot(u_1, u_2) * u_1
        u_2 = tmp / np.linalg.norm(tmp)
        v = np.vstack((u_1, u_2)).T
        l_1 = 0
        l_2 = 0
        tmp_l_1 = e + 1
        tmp_l_2 = e + 1
        while ((abs(tmp_l_1 - l_1) > e) or (abs(tmp_l_2 - l_2) > e)):
            l_1 = tmp_l_1
            l_2 = tmp_l_2
            u = np.dot(A, v)
            B = np.dot(v.T, u)
            ksi = (B[0, 0] - B[1, 1]) / (2 * B[0, 1])
            t = 1 / (2 * ksi)
            c = 1 / np.sqrt(1 + t * t)
            s = c * t
            tmp_l_1 = B[0, 0] * c * c + 2 * B[0, 1] * c * s + B[1, 1] * s * s
            tmp_l_2 = B[0, 0] * s * s - 2 * B[0, 1] * c * s + B[1, 1] * c * c
            u = np.dot(u, np.array([[c, s], [-s, c]]))
            v = MGS(u)[0]
        return ((l_1, l_2))

def Classic_Jacobi(A,e,m,*out):
    if(out):
        tmp_A = A.copy()
        count = 0
        data_1 = np.array([])
        while (np.linalg.norm(tmp_A - np.diag(np.diag(tmp_A))) > e and (count < m)):
            (p, q) = np.unravel_index(np.argmax(np.abs(np.triu(tmp_A, 1))), tmp_A.shape)
            if (abs(tmp_A[p, p] - tmp_A[q, q]) < 1e-14):
                c = np.sqrt(2) / 2
                s = c
            else:
                t = tmp_A[p, q] / (2 * (tmp_A[p, p] - tmp_A[q, q]))
                c = (1 - t * t) / (1 + t * t)
                s = 2 * t / (1 + t * t)
            B = tmp_A.copy()
            for i in range(A.shape[1]):
                if (i != p and i != q):
                    B[i, p] = tmp_A[i, p] * c + tmp_A[i, q] * s
                    B[p, i] = tmp_A[i, p] * c + tmp_A[i, q] * s
                    B[i, q] = -tmp_A[i, p] * s + tmp_A[i, q] * c
                    B[q, i] = -tmp_A[i, p] * s + tmp_A[i, q] * c
            B[p, p] = tmp_A[p, p] * c * c + 2 * tmp_A[p, q] * c * s + tmp_A[q, q] * s * s
            B[q, q] = tmp_A[p, p] * s * s - 2 * tmp_A[p, q] * c * s + tmp_A[q, q] * c * c
            B[p, q] = (tmp_A[q, q] - tmp_A[p, p]) * c * s + tmp_A[p, q] * (c * c - s * s)
            B[q, p] = (tmp_A[q, q] - tmp_A[p, p]) * c * s + tmp_A[p, q] * (c * c - s * s)
            tmp_A = B
            data_1 = np.append(data_1, np.linalg.norm(tmp_A - np.diag(np.diag(tmp_A))))
            count = count + 1
        data = {'IteTimes':np.arange(1,count+1),'error':data_1}
        df = pd.DataFrame(data)
        return (np.diag(tmp_A),df)
    else:
        tmp_A = A.copy()
        count = 0
        while (np.linalg.norm(tmp_A - np.diag(np.diag(tmp_A))) > e and (count < m)):
            (p, q) = np.unravel_index(np.argmax(np.abs(np.triu(tmp_A, 1))), tmp_A.shape)
            if (abs(tmp_A[p, p] - tmp_A[q, q]) < 1e-14):
                c = np.sqrt(2) / 2
                s = c
            else:
                t = tmp_A[p, q] / (2 * (tmp_A[p, p] - tmp_A[q, q]))
                c = (1 - t * t) / (1 + t * t)
                s = 2 * t / (1 + t * t)
            B = tmp_A.copy()
            for i in range(A.shape[1]):
                if (i != p and i != q):
                    B[i, p] = tmp_A[i, p] * c + tmp_A[i, q] * s
                    B[p, i] = tmp_A[i, p] * c + tmp_A[i, q] * s
                    B[i, q] = -tmp_A[i, p] * s + tmp_A[i, q] * c
                    B[q, i] = -tmp_A[i, p] * s + tmp_A[i, q] * c
            B[p, p] = tmp_A[p, p] * c * c + 2 * tmp_A[p, q] * c * s + tmp_A[q, q] * s * s
            B[q, q] = tmp_A[p, p] * s * s - 2 * tmp_A[p, q] * c * s + tmp_A[q, q] * c * c
            B[p, q] = (tmp_A[q, q] - tmp_A[p, p]) * c * s + tmp_A[p, q] * (c * c - s * s)
            B[q, p] = (tmp_A[q, q] - tmp_A[p, p]) * c * s + tmp_A[p, q] * (c * c - s * s)
            tmp_A = B
            count = count + 1
        return np.diag(tmp_A)


def Loop_Jacobi(A,e,m,*out):
    if(out):
        (r, c) = np.shape(A)
        (rows, cols) = np.tril_indices(r, -1, c)
        tmp_A = A.copy()
        count = 0
        anCount = 0
        data_1 = np.array([])
        while (np.linalg.norm(tmp_A - np.diag(np.diag(tmp_A))) > e and (count < m)):
            for (p, q) in zip(rows, cols):
                if (abs(tmp_A[p, p] - tmp_A[q, q]) < 1e-14):
                    c = np.sqrt(2) / 2
                    s = c
                else:
                    t = tmp_A[p, q] / (2 * (tmp_A[p, p] - tmp_A[q, q]))
                    c = (1 - t * t) / (1 + t * t)
                    s = 2 * t / (1 + t * t)
                B = tmp_A.copy()
                for i in range(A.shape[1]):
                    if (i != p and i != q):
                        B[i, p] = tmp_A[i, p] * c + tmp_A[i, q] * s
                        B[p, i] = tmp_A[i, p] * c + tmp_A[i, q] * s
                        B[i, q] = -tmp_A[i, p] * s + tmp_A[i, q] * c
                        B[q, i] = -tmp_A[i, p] * s + tmp_A[i, q] * c
                B[p, p] = tmp_A[p, p] * c * c + 2 * tmp_A[p, q] * c * s + tmp_A[q, q] * s * s
                B[q, q] = tmp_A[p, p] * s * s - 2 * tmp_A[p, q] * c * s + tmp_A[q, q] * c * c
                B[p, q] = (tmp_A[q, q] - tmp_A[p, p]) * c * s + tmp_A[p, q] * (c * c - s * s)
                B[q, p] = (tmp_A[q, q] - tmp_A[p, p]) * c * s + tmp_A[p, q] * (c * c - s * s)
                tmp_A = B
                data_1 = np.append(data_1, np.linalg.norm(tmp_A - np.diag(np.diag(tmp_A))))
                anCount = anCount + 1
            count = count + 1
        data = {'IteTimes': np.arange(1, anCount + 1), 'error': data_1}
        df = pd.DataFrame(data)
        return (np.diag(tmp_A),df)
    else:
        (r, c) = np.shape(A)
        (rows, cols) = np.tril_indices(r, -1, c)
        tmp_A = A.copy()
        count = 0
        while (np.linalg.norm(tmp_A - np.diag(np.diag(tmp_A))) > e and (count < m)):
            for (p, q) in zip(rows, cols):
                if (abs(tmp_A[p, p] - tmp_A[q, q]) < 1e-14):
                    c = np.sqrt(2) / 2
                    s = c
                else:
                    t = tmp_A[p, q] / (2 * (tmp_A[p, p] - tmp_A[q, q]))
                    c = (1 - t * t) / (1 + t * t)
                    s = 2 * t / (1 + t * t)
                B = tmp_A.copy()
                for i in range(A.shape[1]):
                    if (i != p and i != q):
                        B[i, p] = tmp_A[i, p] * c + tmp_A[i, q] * s
                        B[p, i] = tmp_A[i, p] * c + tmp_A[i, q] * s
                        B[i, q] = -tmp_A[i, p] * s + tmp_A[i, q] * c
                        B[q, i] = -tmp_A[i, p] * s + tmp_A[i, q] * c
                B[p, p] = tmp_A[p, p] * c * c + 2 * tmp_A[p, q] * c * s + tmp_A[q, q] * s * s
                B[q, q] = tmp_A[p, p] * s * s - 2 * tmp_A[p, q] * c * s + tmp_A[q, q] * c * c
                B[p, q] = (tmp_A[q, q] - tmp_A[p, p]) * c * s + tmp_A[p, q] * (c * c - s * s)
                B[q, p] = (tmp_A[q, q] - tmp_A[p, p]) * c * s + tmp_A[p, q] * (c * c - s * s)
                tmp_A = B
            count = count + 1
        return np.diag(tmp_A)

def Limit_Loop_Jacobi(A,d,e,m,*out):
    if(out):
        (r, c) = np.shape(A)
        (rows, cols) = np.tril_indices(r, -1, c)
        tmp_A = A.copy()
        count = 0
        anCount = 0
        data_1 = np.array([])
        while (np.linalg.norm(tmp_A - np.diag(np.diag(tmp_A))) > e and (count < m)):
            limit = np.linalg.norm(tmp_A - np.diag(np.diag(tmp_A))) / d
            for (p, q) in zip(rows, cols):
                if (abs(tmp_A[p, q]) >= limit):
                    if (abs(tmp_A[p, p] - tmp_A[q, q]) < 1e-14):
                        c = np.sqrt(2) / 2
                        s = c
                    else:
                        t = tmp_A[p, q] / (2 * (tmp_A[p, p] - tmp_A[q, q]))
                        c = (1 - t * t) / (1 + t * t)
                        s = 2 * t / (1 + t * t)
                    B = tmp_A.copy()
                    for i in range(A.shape[1]):
                        if (i != p and i != q):
                            B[i, p] = tmp_A[i, p] * c + tmp_A[i, q] * s
                            B[p, i] = tmp_A[i, p] * c + tmp_A[i, q] * s
                            B[i, q] = -tmp_A[i, p] * s + tmp_A[i, q] * c
                            B[q, i] = -tmp_A[i, p] * s + tmp_A[i, q] * c
                    B[p, p] = tmp_A[p, p] * c * c + 2 * tmp_A[p, q] * c * s + tmp_A[q, q] * s * s
                    B[q, q] = tmp_A[p, p] * s * s - 2 * tmp_A[p, q] * c * s + tmp_A[q, q] * c * c
                    B[p, q] = (tmp_A[q, q] - tmp_A[p, p]) * c * s + tmp_A[p, q] * (c * c - s * s)
                    B[q, p] = (tmp_A[q, q] - tmp_A[p, p]) * c * s + tmp_A[p, q] * (c * c - s * s)
                    tmp_A = B
                    data_1 = np.append(data_1,np.linalg.norm(tmp_A - np.diag(np.diag(tmp_A))))
                    anCount = anCount + 1
            count = count + 1
        data = {'IteTimes': np.arange(1, anCount + 1), 'error': data_1}
        df = pd.DataFrame(data)
        return (np.diag(tmp_A), df)
    else:
        (r, c) = np.shape(A)
        (rows, cols) = np.tril_indices(r, -1, c)
        tmp_A = A.copy()
        count = 0
        while (np.linalg.norm(tmp_A - np.diag(np.diag(tmp_A))) > e and (count < m)):
            limit = np.linalg.norm(tmp_A - np.diag(np.diag(tmp_A))) / d
            for (p, q) in zip(rows, cols):
                if (abs(tmp_A[p, q]) >= limit):
                    if (abs(tmp_A[p, p] - tmp_A[q, q]) < 1e-14):
                        c = np.sqrt(2) / 2
                        s = c
                    else:
                        t = tmp_A[p, q] / (2 * (tmp_A[p, p] - tmp_A[q, q]))
                        c = (1 - t * t) / (1 + t * t)
                        s = 2 * t / (1 + t * t)
                    B = tmp_A.copy()
                    for i in range(A.shape[1]):
                        if (i != p and i != q):
                            B[i, p] = tmp_A[i, p] * c + tmp_A[i, q] * s
                            B[p, i] = tmp_A[i, p] * c + tmp_A[i, q] * s
                            B[i, q] = -tmp_A[i, p] * s + tmp_A[i, q] * c
                            B[q, i] = -tmp_A[i, p] * s + tmp_A[i, q] * c
                    B[p, p] = tmp_A[p, p] * c * c + 2 * tmp_A[p, q] * c * s + tmp_A[q, q] * s * s
                    B[q, q] = tmp_A[p, p] * s * s - 2 * tmp_A[p, q] * c * s + tmp_A[q, q] * c * c
                    B[p, q] = (tmp_A[q, q] - tmp_A[p, p]) * c * s + tmp_A[p, q] * (c * c - s * s)
                    B[q, p] = (tmp_A[q, q] - tmp_A[p, p]) * c * s + tmp_A[p, q] * (c * c - s * s)
                    tmp_A = B
            count = count + 1
        return np.diag(tmp_A)


def binary_eig(A,e,start,end):

    n = A.shape[1]

    def s(a):
        count = 0
        p_0 = 1
        p_1 = A[0,0] - a
        if(sgn(p_1) > 0):
            count = count + 1
        for i in range(1,n):
            p = (A[i,i] - a) * p_1 - A[i-1,i] * A[i-1,i] * p_0
            if(sgn(p * p_1) > 0):
                count = count + 1
            p_0 = p_1
            p_1 = p
        return count

    while(abs(end - start)>e):
        c = (start + end) / 2
        if(s(c)<s(start)):
            end = c
        else:
            start = c
    return c

if __name__ == '__main__':
    from Defaul_Matrix import T_matrix
    import os
    if (not (os.path.exists(os.getcwd() + '\\6_4result'))):
        os.mkdir(os.getcwd() + '\\6_4result')
    filename = os.getcwd() + '\\6_4result\\'
    '''
    order = np.arange(10,101)
    iteTime = np.array([])
    for i in range(10,101):
        A = T_matrix(i)
        eig = EigOfTmatrix(A)
        res = inv_Power_Method(A, np.ones(i), 1.999, 1e-10, 500, eig[int(i / 2.0)])
        #print(res[2])
        iteTime = np.append(iteTime,np.max(res[2]['IterateTimes']))
    data = {'MatrixOrder': order, 'IterateTimes': iteTime}
    df = pd.DataFrame(data)
    df.to_csv(filename + 'invWithMatrixOrder.csv')
    '''

    #A = T_matrix(100)
    #eig = EigOfTmatrix(A)
    #l_1 = max(eig)
    #res = Power_Method_Aitken(A,np.ones(100),1e-10,50000,l_1)
    #res = Wieldant(A,np.ones(100),1e-10,50000,max(eig))
    #res = simultaneous_iteration(A,1e-10,eig[98],eig[99])
    #res = Limit_Loop_Jacobi(A,101,1e-10,50000,1)
    #res = binary_eig(A,1e-10,1,2)
    #print(res)
    #res = inv_Power_Method(A, np.ones(101), 2.01, 1e-10, 500, eig[50])
    #df = res[2]
    #print(df)
    #fp = open(filename + "Rayleigh101.csv", 'w')
    #df.to_csv(filename + 'Aitken100.csv')
    #plt.rcParams['font.sans-serif'] = ['SimHei']
    #plt.rcParams['axes.unicode_minus'] = False
    #df.plot(x = 'IterateTimes',y = 'Eigvalue',style = '-o')
    #plt.show()
    #print(res)
    #res = inv_Power_Method(A, np.ones(100), res, 1e-10, 500)
    #print(res)
    #print(max(res))
    #Eig = EigOfTmatrix(A)
    #eig = np.transpose(eig[0])
    #eig.sort()
    #Eig.sort()
    #print(eig - Eig)