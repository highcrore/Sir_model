import numpy as np
from scipy.optimize import curve_fit
from scipy.integrate import odeint
import math as m
import matplotlib.pyplot as plt
import pandas as pd
N = 100000000
I0 = 2
R0 = 0
S0 = N-I0-R0
a = 0.2
beta = 0.3

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
x = np.linspace(0,300,187)
y0 = 0
y = odeint(module,y0,x,args=(a,beta))

plt.plot(x,y,color = '#FFA833',linestyle = '-')
plt.show()

