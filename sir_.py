import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import pandas as pd

#original

data = pd.read_csv('6_1.csv')
recovered_covid = data['recovery']
death_covid = data['death']
infected_covid = data['active cases']
ls1 = []
ls2 = []
for i in recovered_covid:
    ls1.append(i)
for j in death_covid:
    ls2.append(j)
recovered = [sum(i) for i in zip(ls1, ls2)]
recovered_1 = []
recovered_2 = []
for i in range(0,31):
    recovered_1.append(recovered[i])
for i in range(31,61):
    recovered_2.append(recovered[i])
if1 = []
INFECTED_1 = []
INFECTED_2 = []
for d in infected_covid:
    if1.append(d)
for i in range(0,31):
    INFECTED_1.append(if1[i])
for i in range(31,61):
    INFECTED_2.append(if1[i])



#lineralgebra

#aug
N = 100000000 #population
I0 = 150662
R0 = 256158+15298
S0 = N-I0-R0

a1 = 0.07595083
beta1 = 0.06617994

t1 = np.linspace(0,30,31)

def module1(z,t, n, a,beta):
    S = z[0]
    I = z[1]
    R = z[2]
    ds_dt = ((-1)*beta*S*I)/n
    di_dt = ((beta*S*I)/n) - a*I
    dr_dt = a*I

    return [ds_dt,di_dt,dr_dt]


z01 = [S0,I0,R0]
z1 = odeint(module1, z01, t1, args=(N, a1, beta1))

Sl1 = z1[:,0]
Il1 = z1[:,1]
Rl1 = z1[:,2]

#sept
N = 100000000 #population
I02 = 194056
R02 = 573559+24926
S02 = N-I0-R0

a2 = 0.07666238
beta2 = 0.06655908

t2 = np.linspace(31, 60, 30)

def module2(z,t, n, a,beta):
    S = z[0]
    I = z[1]
    R = z[2]
    ds_dt = ((-1)*beta*S*I)/n
    di_dt = ((beta*S*I)/n) - a*I
    dr_dt = a*I

    return [ds_dt,di_dt,dr_dt]


z02 = [S02,I02,R02]
z2 = odeint(module2, z02, t2, args=(N, a2, beta2))

Sl2 = z2[:,0]
Il2 = z2[:,1]
Rl2 = z2[:,2]


#numerical

#aug
N = 100000000 #population
I0 = 150662
R0 = 256158+15298
S0 = N-I0-R0

a1 = 0.06982
beta1 = 0.07094

t1 = np.linspace(0,30,31)

def module1(z,t, n, a,beta):
    S = z[0]
    I = z[1]
    R = z[2]
    ds_dt = ((-1)*beta*S*I)/n
    di_dt = ((beta*S*I)/n) - a*I
    dr_dt = a*I

    return [ds_dt,di_dt,dr_dt]


z01 = [S0,I0,R0]
z1 = odeint(module1, z01, t1, args=(N, a1, beta1))

SN1 = z1[:,0]
IN1 = z1[:,1]
RN1 = z1[:,2]

#sept
N = 100000000 #population
I02 = 194056
R02 = 573559+24926
S02 = N-I0-R0

a2 = 0.0871
beta2 = 0.1104

t2 = np.linspace(31, 60, 30)

def module2(z,t, n, a,beta):
    S = z[0]
    I = z[1]
    R = z[2]
    ds_dt = ((-1)*beta*S*I)/n
    di_dt = ((beta*S*I)/n) - a*I
    dr_dt = a*I

    return [ds_dt,di_dt,dr_dt]


z02 = [S02,I02,R02]
z2 = odeint(module2, z02, t2, args=(N, a2, beta2))

SN2 = z2[:,0]
IN2 = z2[:,1]
RN2 = z2[:,2]


#analytical

#aug
N = 100000000 #population
I0 = 150662
R0 = 256158+15298
S0 = N-I0-R0

a1 = 0.23594448858
beta1 = 0.04349283

t1 = np.linspace(0,30,31)

def module1(z,t, n, a,beta):
    S = z[0]
    I = z[1]
    R = z[2]
    ds_dt = ((-1)*beta*S*I)/n
    di_dt = ((beta*S*I)/n) - a*I
    dr_dt = a*I

    return [ds_dt,di_dt,dr_dt]


z01 = [S0,I0,R0]
z1 = odeint(module1, z01, t1, args=(N, a1, beta1))

Sa1 = z1[:,0]
Ia1 = z1[:,1]
Ra1 = z1[:,2]

#sept
N = 100000000 #population
I02 = 194056
R02 = 573559+24926
S02 = N-I0-R0

a2 = 0.0231208109
beta2 = 0.0238026056

t2 = np.linspace(31, 60, 30)

def module2(z,t, n, a,beta):
    S = z[0]
    I = z[1]
    R = z[2]
    ds_dt = ((-1)*beta*S*I)/n
    di_dt = ((beta*S*I)/n) - a*I
    dr_dt = a*I

    return [ds_dt,di_dt,dr_dt]


z02 = [S02,I02,R02]
z2 = odeint(module2, z02, t2, args=(N, a2, beta2))

Sa2 = z2[:,0]
Ia2 = z2[:,1]
Ra2 = z2[:,2]



x1 = np.linspace(0,30,31)
x2 = np.linspace(31,60,30)


# plt.plot(x1,IN1,color = '#FF0800',label = 'numerical')
# plt.plot(x2,IN2,'o',markersize=1,color = '#FF0800')
# plt.plot(x1,Il1,color = '#000000',label = 'linear')
# plt.plot(x2,Il2,'o',markersize=1,color = '#000000')
plt.plot(x1,RN1,color = '#38E232',label = 'numerical')
# plt.plot(x2,RN2,'o',markersize=1,color = '#38E232' )
plt.plot(x1,Rl1,color = '#D0E232',label = 'linear')
# plt.plot(x2,Rl2,'o',markersize=1,color = '#D0E232')
# plt.plot(x1,Ia1,color = '#E232E2',label = 'analytical')
#plt.plot(x2,Ia2,'o',markersize=1,color = '#E232E2')
plt.plot(x1,recovered_1,label = 'data',color = '#E26732')
# plt.plot(x2,recovered_2,'o',markersize=1,color = '#E26732')
# plt.plot(x1,INFECTED_1,color = '#3332E2',label = 'data')
# plt.plot(x2,INFECTED_2,'o',markersize=1,color = '#3332E2')
plt.xlabel('no of days')

plt.legend()
plt.show()

