import numpy as np
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt
from scipy.integrate import odeint
import math as m
import csv
import pandas as pd
import warnings
from array import *
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
for i in range(84):
    recovered_1.append(recovered[i])
for i in range(84,187):
    recovered_2.append(recovered[i])
major = []
major.append(recovered_1)
major.append(recovered_2)
I = [[2.0]]
R = [[0.0]]
c = len(I)
d = len(R)
S_values = []
I_values = []
R_values = []
def SIR(array_2d):

    new_len = []
    length = len(array_2d)
    for i in range(length):
        def inner_sir(rec_list):
            global c,d
            def a_beta(rec_list):

                def model(x, a, beta):
                    row = (a * N) / beta
                    alpha = m.sqrt(((S0 / row) - 1) ** 2 + (2 * S0 * (N - S0)) / (row ** 2))
                    phai = m.atanh((1 / alpha) * ((S0 / row) - 1))
                    rx = ((row ** 2) / S0) * (((S0 / row) - 1) + alpha * (np.tanh(((a * alpha * x) / 2) - phai)))
                    return rx
                new_len.append(len(rec_list))

                if i == 0:
                    x = np.linspace(0, len(rec_list), len(rec_list))
                elif i >0:
                    x = np.linspace(85, 85+len(rec_list), len(rec_list))
                init_guess = [0.2, 0.25]
                param, param_cov = curve_fit(model, x, rec_list, p0=init_guess, absolute_sigma=True)

                gama, beta = param
                print(gama)
                print(beta)
                return gama, beta
            def module(z, t, n, a, beta):
                S = z[0]
                I = z[1]
                R = z[2]
                ds_dt = ((-1) * beta * S * I) / n
                di_dt = ((beta * S * I) / n) - a * I
                dr_dt = a * I

                return [ds_dt, di_dt, dr_dt]

            N = 100000000
            I0 = I[c - 1][-1]
            R0 = R[d-1][-1]

            S0 = N - I0 - R0
            a, beta = a_beta(rec_list)

            if i == 0:
                t = np.linspace(0, len(rec_list), len(rec_list)+1)
            elif i == 1:
                t = np.linspace(85, 85+len(rec_list), len(rec_list)+1)
            z0 = [S0, I0, R0]
            z = odeint(module, z0, t, args=(N, a, beta))

            S1 = z[:, 0]
            I1 = z[:, 1]
            R1 = z[:, 2]
            I.append(I1)
            R.append(R1)
            c = len(I)
            d = len(R)
            return S1,I1,R1
        Sf,If,Rf = inner_sir(array_2d[i])
        S_values.append(Sf)
        I_values.append(If)
        R_values.append(Rf)
SIR(major)

x = np.linspace(0,84,85)
x1 = np.linspace(85,188,104)
plt.plot(x,I_values[0],color ='#0012FF',ls = ':',markersize = 1,label = 'actual')
plt.plot(x1,I_values[1],color ='r',ls = ':',markersize = 1,label = 'actual1')
# # plt.plot(t,R,color = '#57594F',label= 'by SIR')
# # plt.xlabel('number of days')
# # plt.ylabel('people(in hundred thousand)')
# # plt.title('recovered actual/recovered by model')
# # # ya = [200000,400000,600000,800000,1000000]
# # # label = ['2','4','6','8','10']
# # # plt.yticks(ya,label)
plt.legend()
plt.show()