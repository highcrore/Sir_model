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
data = pd.read_csv('covied_data.csv')
recovered_covid = data['Recovered']
death_covid = data['Deceased']
ls1 = []
ls2 = []
for i in recovered_covid:
    ls1.append(i)
for j in death_covid:
    ls2.append(j)
recovered = [sum(i) for i in zip(ls1, ls2)]


def model(x,a,beta):


    def module(r,t):
        N = 100000000
        I0 = 2
        R0 = 0
        S0 = N - I0 - R0
        row = (a * N) / beta
        alpha = m.sqrt(((S0 / row) - 1) ** 2 + (2 * S0 * (N - S0)) / (row ** 2))
        phai = m.atanh((1 / alpha) * ((S0 / row) - 1))
        dr_dx = ((a * (alpha ** 2) * (row ** 2)) / (2 * S0))*(1/(m.cosh(((alpha*a*t)/2)-phai))**2)
        return dr_dx
    r0 = 0
    t = np.linspace(0,186,187)
    res = odeint(module,r0,t)
    resn = np.array(res)
    d = resn.ravel()

    return d

x = np.linspace(0,186,187)
init_guess = [0.2,0.25]
param,param_cov = curve_fit(model,x,recovered,p0=init_guess,absolute_sigma=True)

aa, betaa = param
print(aa)
print(betaa)

def module(y,x,a,beta):
    N = 100000000
    I0 = 2
    R0 = 0
    S0 = N - I0 - R0
    row = (a * N) / beta
    alpha = m.sqrt(((S0 / row) - 1) ** 2 + (2 * S0 * (N - S0)) / (row ** 2))
    phai = m.atanh((1 / alpha) * ((S0 / row) - 1))
    dr_dx = ((a*(alpha**2)*(row**2))/(2*S0))*((1/(m.cosh(((alpha*a*x)/2)-phai))**2))

    return dr_dx
x1 = np.linspace(0,300,187)
y0 = 0
y = odeint(module,y0,x1,args=(aa,betaa))





plt.plot(x,recovered,color ='#0012FF',ls = ':',markersize = 1,label = 'actual')
plt.plot(x1,y,color = '#57594F',label= 'by SIR')
plt.xlabel('number of days')
plt.ylabel('people(in hundred thousand)')
plt.title('recovered actual/recovered by model')
ya = [200000,400000,600000,800000,1000000]
label = ['2','4','6','8','10']
plt.yticks(ya,label)
plt.legend()
plt.show()