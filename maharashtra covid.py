import pandas as pd
import csv
from matplotlib import pyplot as plt
import numpy as np
number_of_days = [i for i in range(0,187)]
population_maharashtra = 100000000

# Suspected = N-I-R
# where I = infected, R = recovered+dead




data = pd.read_csv('covied_data.csv')
infected_covid = data['Confirmed']
recovered_covid = data['Recovered']
death_covid = data['Deceased']

#lists to store data to extracted from csv
ls1 = []
ls2 = []
infected = []

for i in recovered_covid:
    ls1.append(i)
for j in death_covid:
    ls2.append(j)
for g in infected_covid:
    infected.append(g)

recovered = [sum(i) for i in zip(ls1, ls2)]

#suspected = population-infected-recovered

suspected = []
for i in range(187):
    suspected.append(population_maharashtra-(infected[i]+recovered[i]))









plt.plot(number_of_days,suspected,color = '#FF0800',label = 'suspected')
plt.plot(number_of_days,infected,'o',markersize=1,color = '#000000',label = 'infected')
plt.plot(number_of_days,recovered,color = '#020EA2',label = 'recovered')

plt.title('maharashtra covid data')
plt.xlabel('no of days')
plt.ylabel('number of person(million)')
plt.legend()
plt.show()
