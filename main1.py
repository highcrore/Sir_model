import numpy as np
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt
from scipy.integrate import odeint
import math as m
import csv
import pandas as pd
import warnings
plt.figure(figsize = (5,4),dpi=200)
warnings.filterwarnings('ignore')
data = pd.read_csv('covid_1.csv')
recovered_covid = data['Recovered']
death_covid = data['Deceased']
ls1 = []
ls2 = []
for i in recovered_covid:
    ls1.append(i)
for j in death_covid:
    ls2.append(j)
recovered = [sum(i) for i in zip(ls1, ls2)]
recovered_1 = []
recovered_2 = []

for l in range(84):
    recovered_1.append(recovered[l])
for k in range(84,187):
    recovered_2.append((recovered[k]))

#lockdown
def model(t, a, beta):
    N = 100000000
    I0 = 2.0
    R0 = 0.0
    S0 = N-I0-R0
    row = (a * N) / beta
    alpha = m.sqrt(((S0 / row) - 1) ** 2 + (2 * S0 * (N - S0)) / (row ** 2))
    phai = m.atanh((1 / alpha) * ((S0 / row) - 1))
    rt = ((row ** 2) / S0) * (((S0 / row) - 1) + alpha * (np.tanh(((a * alpha * t) / 2) - phai)))
    return rt


t1 = np.linspace(0, 83, 84)
init_guess = [0.2, 0.25]
param, param_cov = curve_fit(model, t1, recovered_1, p0=init_guess, absolute_sigma=True)
a, beta = param
print(a)
print(beta)

def module(z, t, n, a, beta):
    S = z[0]
    I = z[1]
    R = z[2]
    ds_dt = ((-1) * beta * S * I) / n
    di_dt = ((beta * S * I) / n) - a * I
    dr_dt = a * I

    return [ds_dt, di_dt, dr_dt]
N = 100000000
I0 = 2.0
R0 = 0.0
S0 = N-I0-R0
z0 = [S0, I0, R0]
t_1 = np.linspace(0, 83, 84)
z = odeint(module, z0, t_1, args=(N, a, beta))
S1 = z[:, 0]
I1 = z[:, 1]
R1 = z[:, 2]
I0_new = I1[-1]
R0_new = R1[-1]

#unlock
def model1(t, a, beta):

    row = (a * N) / beta
    alpha = m.sqrt(((S0_1 / row) - 1) ** 2 + (2 * S0_1 * (N - S0_1)) / (row ** 2))
    phai = m.atanh((1 / alpha) * ((S0_1 / row) - 1))
    rt = ((row ** 2) / S0_1) * (((S0_1 / row) - 1) + alpha * (np.tanh(((a * alpha * t) / 2) - phai)))
    return rt


t2 = np.linspace(84, 187, 103)
init_guess = [0.5, 0.5]
N = 100000000
I0_1 = I0_new
R0_1 = R0_new
S0_1 = N - I0_1 - R0_1
param1, param_cov1 = curve_fit(model1, t2, recovered_2, p0=init_guess, absolute_sigma=True)
a1, beta1 = param1
print('a new',a1)
print(beta1)

def module(z, t, n, a, beta):
    S = z[0]
    I = z[1]
    R = z[2]
    ds_dt = ((-1) * beta * S * I) / n
    di_dt = ((beta * S * I) / n) - a * I
    dr_dt = a * I

    return [ds_dt, di_dt, dr_dt]

z0 = [S0_1, I0_1, R0_1]
t_2 = np.linspace(84, 187, 103)
z_new = odeint(module, z0, t_2, args=(N, a1, beta1))
S1_new = z_new[:, 0]
I1_new = z_new[:, 1]
R1_new = z_new[:, 2]




x = np.linspace(0,83,84)
x1 = np.linspace(0,83,84)
plt.plot(x,R1,color ='#0012FF',ls = ':',markersize = 1,label = 'sir recovered')
plt.plot(x1,recovered_1,color ='r',ls = ':',markersize = 1,label = 'actual -recovered')
#plt.plot(x2,recovered,color ='y',ls = '--',markersize = 1,label = 'recovered')

x_1= np.linspace(84, 187, 103)
x_2 = np.linspace(84, 187, 103)
plt.plot(x_1,R1_new,ls = ':',markersize = 1,label = 'sir recovered2')
plt.plot(x_2,recovered_2,ls = ':',markersize = 1,label = 'actual -recovered2')




plt.legend()
plt.show()