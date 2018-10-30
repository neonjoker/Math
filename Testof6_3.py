from Defaul_Matrix import T_matrix
import numpy as np
from QU import MGS
from QU import CGS
from QU import Householder
from QU import Givens
sgn = lambda x: 1 if x > 0 else -1 if x < 0 else 0

def Generate_Question(n):
    G = T_matrix(n,n-1)
    b = np.zeros(n)
    for k in range(n-1):
        b[k] = (k+1)/n
    b[0] = b[0] + 1
    b[n-2] = b[n-2] + 1
    b[n-1] = 0
    times = np.random.randint(10,21)
    for k in range(times):
        sin = np.random.rand()
        cos = np.sqrt(1 - sin * sin)
        i = np.random.randint(0, n)
        j = np.random.randint(0, n)
        while (j == i):
            j = np.random.randint(0, n)
        (tmp_1,tmp_2) = (b[j],b[i])
        b[j] = cos * tmp_1 + sin * tmp_2
        b[i] = cos * tmp_2 - sin * tmp_1
        for c in range(n-1):
            (tmp_1, tmp_2) = (G[j, c], G[i, c])
            G[j, c] = cos * tmp_1 + sin * tmp_2
            G[i, c] = cos * tmp_2 - sin * tmp_1
    return (G,b)

def solve_L(L,b):
    n = L.shape[0]
    x = np.zeros(n)
    x[0] = b[0] / L[0,0]
    for i in range(1,n):
        x[i] = b[i]
        for j in range(i):
            x[i] = x[i] - L[i,j] * x[j]
        x[i] = x[i] / L[i,i]
    return x

def solve_U(U,b):
    n = U.shape[0]
    x = np.zeros(n)
    x[n-1] = b[n-1] / U[n-1,n-1]
    for i in range(n-2,-1,-1):
        x[i] = b[i]
        for j in range(i+1,n):
            x[i] = x[i] - U[i,j] * x[j]
        x[i] = x[i] / U[i,i]
    return x

def Householder_solve(A,b):
    (r, c) = np.shape(A)
    qb = b.copy()
    #Q = np.identity(r)
    R = np.copy(A)
    s = min(r,c)
    if (c < r):
        s = s + 1
    D = np.zeros(s - 1)
    for cnt in range(s - 1):
        D[cnt] = np.linalg.norm(R[cnt:,cnt])
        R[cnt,cnt] = (R[cnt,cnt] - D[cnt])
        R[cnt:,cnt] = R[cnt:,cnt] / np.linalg.norm(R[cnt:,cnt])
        qb[cnt:] = qb[cnt:] - 2 * np.dot(R[cnt:,cnt].T,qb[cnt:]) * R[cnt:,cnt]
        for i in range(cnt+1,c):
            R[cnt:,i] = R[cnt:,i] - 2 * np.dot(R[cnt:,cnt].T,R[cnt:,i]) * R[cnt:,cnt]

    for i in range(s-1):
        R[i,i] = D[i]
    R = np.triu(R)
    s = min(r,c)
    x = solve_U(R[:s, :s], qb)
    return x

def Givens_solve(A,b):
    (r, c) = np.shape(A)
    s = min(r,c)
    if(c<r):
        s = s + 1
    qb = b.copy()
    #Q = np.identity(r)
    R = np.copy(A)
    for col in range(s - 1):
        for row in range(col+1 ,r):
            if (R[row, col] != 0):
                r_ = np.hypot(R[col, col], R[row, col])  # d
                cos = R[col, col] / r_
                sin = -R[row, col] / r_
                rho = 1
                if(cos != 0):
                    if(abs(sin)<abs(cos)):
                        rho = sgn(cos) * sin / 2
                    else:
                        rho = 2 * sgn(sin) / cos
                (R[col,col],R[row,col]) = (r_,rho)
                (tmp_1,tmp_2) = (qb[col],qb[row])
                qb[col] = cos * tmp_1 - sin * tmp_2
                qb[row] = cos * tmp_2 + sin * tmp_1
                for i in range(col+1,c):
                    (a, b) = (R[col, i], R[row, i])
                    R[col, i] = cos * a - sin * b
                    R[row, i] = cos * b + sin * a
    R = np.triu(R)
    s = min(r,c)
    x = solve_U(R[:s,:s],qb)
    return x

if __name__ == '__main__':

    import time
    import os
    import csv

    from Defaul_Matrix import cond_bad_matrix
    from Defaul_Matrix import Givens_RandomMultiply
    A = np.dot(np.dot(Givens_RandomMultiply(80,100),cond_bad_matrix(80)),Givens_RandomMultiply(80,100))
    R1 = CGS(A)[1]
    R2 = MGS(A)[1]
    O = cond_bad_matrix(80)
    out = open('C:\\Math\\6_3result\\CGSandMGS.csv', 'w', newline='')
    csv_write = csv.writer(out,dialect='excel')
    csv_write.writerow(['i','d','type'])
    for i in range(80):
        csv_write.writerow([i+1,R1[i,i],'CGS'])
        csv_write.writerow([i + 1, R2[i, i], 'MGS'])
        csv_write.writerow([i + 1, O[i, i], 'ORIGINAL'])

'''
    if not (os.path.exists(os.getcwd() + '\\6_3result')):
        os.mkdir(os.getcwd() + '\\6_3result')
    filename = os.getcwd() + '\\6_3result\\'

    out = open(filename + 'Basic.csv', 'w', newline='')
    csv_write = csv.writer(out, dialect='excel')
    csv_write.writerow(['矩阵阶数', 'CPU时间', '误差'])
    out = open(filename + 'MGS.csv', 'w', newline='')
    csv_write = csv.writer(out, dialect='excel')
    csv_write.writerow(['矩阵阶数', 'CPU时间', '误差'])
    out = open(filename + 'Householder.csv', 'w', newline='')
    csv_write = csv.writer(out, dialect='excel')
    csv_write.writerow(['矩阵阶数', 'CPU时间', '误差'])
    out = open(filename + 'Givens.csv', 'w', newline='')
    csv_write = csv.writer(out, dialect='excel')
    csv_write.writerow(['矩阵阶数', 'CPU时间', '误差'])

    for i in range(100, 501):
        Question = Generate_Question(i)
        A = Question[0]
        b = Question[1]
        r_x = np.ones((i - 1))
        # 法方程组

        start = time.perf_counter()
        L = np.linalg.cholesky(np.dot(A.T, A))
        y = solve_L(L, np.dot(A.T, b))
        x_1 = solve_U(L.T, y)
        end = time.perf_counter()
        t_1 = end - start
        e_1 = np.linalg.norm(x_1 - r_x)
        out = open(filename + 'Basic.csv', 'a', newline='')
        csv_write = csv.writer(out, dialect='excel')
        csv_write.writerow([i, t_1, e_1])
        # print('法方程组解为:')
        # print(x_1)
        # print('耗时：', end='')
        # print(t_1)

        # MGS
        start = time.perf_counter()
        res = MGS(A)
        x_2 = solve_U(res[1], np.dot(res[0].T, b))
        end = time.perf_counter()
        t_2 = end - start
        e_2 = np.linalg.norm(x_2 - r_x)
        out = open(filename + 'MGS.csv', 'a', newline='')
        csv_write = csv.writer(out, dialect='excel')
        csv_write.writerow([i, t_2, e_2])
        # print('MGS解为:')
        # print(x_2)
        # print('耗时：', end='')
        # print(t_2)

        # Householder
        start = time.perf_counter()
        x_3 = Householder_solve(A, b)
        end = time.perf_counter()
        t_3 = end - start
        e_3 = np.linalg.norm(x_3 - r_x)
        out = open(filename + 'Householder.csv', 'a', newline='')
        csv_write = csv.writer(out, dialect='excel')
        csv_write.writerow([i, t_3, e_3])
        # print('Householder解为:')
        # print(x_3)
        # print('耗时：', end='')
        # print(t_3)

        # Givens
        start = time.perf_counter()
        x_4 = Givens_solve(A, b)
        end = time.perf_counter()
        t_4 = end - start
        e_4 = np.linalg.norm(x_4 - r_x)
        out = open(filename + 'Givens.csv', 'a', newline='')
        csv_write = csv.writer(out, dialect='excel')
        csv_write.writerow([i, t_4, e_4])
        # print('Givens解为:')
        # print(x_4)
        # print('耗时：', end='')
        # print(t_4)

    out = open(filename + 'CGS_QR.csv', 'w', newline='')
    csv_write = csv.writer(out, dialect='excel')
    csv_write.writerow(['矩阵阶数', 'CPU时间', '正交性', '向后稳定性'])
    out = open(filename + 'MGS_QR.csv', 'w', newline='')
    csv_write = csv.writer(out, dialect='excel')
    csv_write.writerow(['矩阵阶数', 'CPU时间', '正交性', '向后稳定性'])
    out = open(filename + 'Householder_QR.csv', 'w', newline='')
    csv_write = csv.writer(out, dialect='excel')
    csv_write.writerow(['矩阵阶数', 'CPU时间', '正交性', '向后稳定性'])
    out = open(filename + 'Givens_QR.csv', 'w', newline='')
    csv_write = csv.writer(out, dialect='excel')
    csv_write.writerow(['矩阵阶数', 'CPU时间', '正交性', '向后稳定性'])
    for k in range(20, 31):
        A = np.random.randint(3, 10) * np.random.rand(k, k)
        while (np.linalg.det(A) < 0):
            A = np.random.randint(3, 10) * np.random.rand(k, k)
        # CGS
        start = time.perf_counter()
        res = CGS(A)
        end = time.perf_counter()
        t = end - start
        Q = res[0]
        R = res[1]
        I = np.dot(Q.T, Q)
        i = np.identity(I.shape[0])
        e_i = np.linalg.norm(I - i)
        e = np.linalg.norm(A - np.dot(Q, R)) / np.linalg.norm(A)
        out = open(filename + 'CGS_QR.csv', 'a', newline='')
        csv_write = csv.writer(out, dialect='excel')
        csv_write.writerow([k, t, e_i, e])
        # MGS
        start = time.perf_counter()
        res = MGS(A)
        end = time.perf_counter()
        t = end - start
        Q = res[0]
        R = res[1]
        I = np.dot(Q.T, Q)
        i = np.identity(I.shape[0])
        e_i = np.linalg.norm(I - i)
        e = np.linalg.norm(A - np.dot(Q, R)) / np.linalg.norm(A)
        out = open(filename + 'MGS_QR.csv', 'a', newline='')
        csv_write = csv.writer(out, dialect='excel')
        csv_write.writerow([k, t, e_i, e])
        # Householder
        start = time.perf_counter()
        res = Householder(A)
        end = time.perf_counter()
        t = end - start
        Q = res[0]
        R = res[1]
        I = np.dot(Q.T, Q)
        i = np.identity(I.shape[0])
        e_i = np.linalg.norm(I - i)
        e = np.linalg.norm(A - np.dot(Q, R)) / np.linalg.norm(A)
        out = open(filename + 'Householder_QR.csv', 'a', newline='')
        csv_write = csv.writer(out, dialect='excel')
        csv_write.writerow([k, t, e_i, e])
        # Givens
        start = time.perf_counter()
        res = Givens(A)
        end = time.perf_counter()
        t = end - start
        Q = res[0]
        R = res[1]
        I = np.dot(Q.T, Q)
        i = np.identity(I.shape[0])
        e_i = np.linalg.norm(I - i)
        e = np.linalg.norm(A - np.dot(Q, R)) / np.linalg.norm(A)
        out = open(filename + 'Givens_QR.csv', 'a', newline='')
        csv_write = csv.writer(out, dialect='excel')
        csv_write.writerow([k, t, e_i, e])
'''

