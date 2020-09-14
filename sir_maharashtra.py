import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

N = 100000000 #population
I0 = 2.0
R0 = 0
S0 = N-I0-R0

# r = Transmissibility of disease
# 1/a = average recovery time
# order of r is of N, so: r = beta/N
# we a(ssume Ro to be 2.5, so we get beta = 0.5

a = 0.2
beta = 0.5

t = np.linspace(0,187,373)

def module(z,t, n, a,beta):
    S = z[0]
    I = z[1]
    R = z[2]
    ds_dt = ((-1)*beta*S*I)/n
    di_dt = ((beta*S*I)/n) - a*I
    dr_dt = a*I

    return [ds_dt,di_dt,dr_dt]


z0 = [S0,I0,R0]
z = odeint(module, z0, t, args=(N, a, beta))

S = z[:,0]
I = z[:,1]
R = z[:,2]

plt.plot(t,S,color = '#FF0800',label = 'suspected')
plt.plot(t,I,'o',markersize=1,color = '#000000',label = 'infected')
plt.plot(t,R,color = '#020EA2',label = 'recovered')
plt.xlabel('no of days')

plt.legend()
plt.show()
