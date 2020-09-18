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
# we are extracting recovered data from csv file
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
recovered = [sum(i) for i in zip(ls1, ls2)]  # recovered = recovered +death
recovered_1 = []
recovered_2 = []
recovered_3 = []
recovered_4 = []

for i in range(37):                            # created separate list for each phase lockdown
    recovered_1.append(recovered[i])
for i in range(37,56):
    recovered_2.append(recovered[i])
for i in range(56,70):
    recovered_3.append(recovered[i])
for i in range(70,84):
    recovered_4.append(recovered[i])

major = [recovered_1, recovered_2, recovered_3,recovered_4]  # added all recovered list in major 2d list which will be called in main function
I = [[2.0]]  #initial values
R = [[0.0]]
c = len(I) # c,d have used in function to get last element of I,R TO USE IT AS I0,R0 in next phase lockdown
d = len(R)
S_values = []    #Final result will be saved in this list for plotting
I_values = []
R_values = []


def SIR(array_2d):

    length = len(array_2d)
    for i in range(length):   # length is number of recovered list available
        if i == 0:
            x = np.linspace(0, 36, 37)
        elif i == 1:
            x = np.linspace(37,55, 19)
        elif i == 2:
            x = np.linspace(56, 69, 14)
        elif i == 3:
            x = np.linspace(70, 83, 14)
        def inner_sir(rec_list):
            global c,d
            def a_beta(rec_list):  #function to give curve fit parameter a, beta

                def model(x, a, beta):   #function to give curve fit parameter a, beta
                    row = (a * N) / beta
                    alpha = m.sqrt(((S0 / row) - 1) ** 2 + (2 * S0 * (N - S0)) / (row ** 2))
                    phai = m.atanh((1 / alpha) * ((S0 / row) - 1))
                    rx = ((row ** 2) / S0) * (((S0 / row) - 1) + alpha * (np.tanh(((a * alpha * x) / 2) - phai)))
                    return rx



                init_guess = [0.2, 0.25]
                param, param_cov = curve_fit(model, x, rec_list, p0=init_guess, absolute_sigma=True)

                a, beta = param
                # if i ==1 or i==3:
                #     beta = -beta*10
                print('a',a)
                print('beta',beta)
                return a, beta
            def module(z, t, n, a, beta):  #uses a,beta and give values of SIR
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
            a, beta = a_beta(rec_list) #function has been called

            if i == 0:
                t = np.linspace(0, 36, 37)
            elif i == 1:
                t = np.linspace(37, 55, 19)
            elif i == 2:
                t = np.linspace(56, 69, 14)
            elif i == 3:
                t = np.linspace(70, 83, 14)
            z0 = [S0, I0, R0]

            z = odeint(module, z0, t, args=(N, a, beta))  #integrator function has been called

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

# x = np.linspace(0,36,37)
# x1 = np.linspace(37,55,19)
# x2 = np.linspace(56,69,14)
# x3 = np.linspace(70,83,14)
# plt.plot(x,I_values[0],color ='#0012FF',ls = '-',markersize = 1,label = 'lockdown_phase1')
# plt.plot(x1,I_values[1],color ='#FF0101',ls = '-',markersize = 1,label = 'lockdown_phase2')
# plt.plot(x2,I_values[2],color ='#1C457B',ls = '-',markersize = 1,label = 'lockdown_phase3')
# plt.plot(x3,I_values[3],color ='#7B1C37',ls = '-',markersize = 1,label = 'lockdown_phase4')
# # # plt.plot(t,R,color = '#57594F',label= 'by SIR')
# # # plt.xlabel('number of days')
# # # plt.ylabel('people(in hundred thousand)')
# # # plt.title('recovered actual/recovered by model')
# # # # ya = [200000,400000,600000,800000,1000000]
# # # # label = ['2','4','6','8','10']
# # # # plt.yticks(ya,label)
# plt.legend()
# plt.show()