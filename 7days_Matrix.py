import numpy as np
import pandas as pd

'''  Rate of change of susceptibles: ds/dt = -bs(t)I(t)
where s(t) = S(t)/N
Rate of change of the number of recovered: dR/dt = kI(t)
rate of change of the number of the infected: dI/dt = bs(t)I(t) − kI(t)
'''

data = pd.read_csv('october.csv')
ls1 = data['active cases']
ls2 = data['recovery']
ls3 = data['death']
recovered = [sum(i) for i in zip(ls2, ls3)]
infected = []
for i in ls1:
    infected.append(i)

# creating sj,Ij,Rj

N = 100000000  # Population
s = []
for i in range(8):
    It = infected[i]
    Rt = recovered[i]
    st = (N - It - Rt) / N
    s.append(st)
I = infected
R = recovered
# creating cj1 ,cj2 and uj

cj1 = []
cj2 = []
for j in range(1, 8):
    ct1 = s[j - 1] * I[j - 1]
    cj1.append(ct1)
    ct2 = (-1) * I[j - 1]
    cj2.append(ct2)
uj = []
for j in range(1, 8):
    ut = I[j] - I[j - 1]
    uj.append(ut)


# creating C and u

cjlist = [cj1, cj2]
cjarray = np.array(cjlist)
Cmatrix = np.matrix(cjarray)  # CREATED matrix of cj1,cj2
C = Cmatrix.T  # matrix transpose is taken, now shape of matrix is (7,2)
uarr = np.array(uj)
umatrix = np.matrix(uarr)
u = umatrix.T  # u is matrix of uj

# Creating dj1, dj2, vj

dj1 = [0]*7
dj2 = []
for j in range(1,8):
    dt = I[j-1]
    dj2.append(dt)
vj = []
for j in range(1,8):
    vt = R[j]-R[j-1]
    vj.append(vt)

# Creating D, v

djlist = [dj1,dj2]
djarray = np.array(djlist)
djmatrix = np.matrix(djarray)
D = djmatrix.T

varr = np.array(vj)
vmatrix = np.matrix(varr)
v = vmatrix.T

# creating A and w

m1 = Cmatrix
m2 = djmatrix
m11 = m1.T
m22 = m2.T
A = np.vstack((m11,m22))
w = np.vstack((u,v))
A_tran = A.transpose()
temp_result = np.dot(A_tran,A)
temp_result = np.linalg.inv(temp_result)
temp_result= np.dot(temp_result,A_tran)
result = np.dot(temp_result,w)
print(result)


