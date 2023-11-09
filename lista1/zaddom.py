import matplotlib.pyplot as plt
import numpy as np

M = 1000
amount = 1000
mi = 1
sigma = 0.5
l_mnw =[]
l_mm = []

for i in range(M):    
    normal =np.random.normal(mi,sigma,amount)
    lognormal =np.exp(normal)

    mi_mnw = 1/amount*sum(np.log(lognormal))
    m1 = 1/amount*sum(lognormal)
    m2 = 1/amount*sum(lognormal**2)

    l_mnw.append(1/amount*sum((np.log(lognormal)-mi_mnw)**2))
    l_mm.append(np.log(m2)-2*np.log(m1))


plt.boxplot([l_mnw,l_mm])
plt.title("Boxplots")
plt.legend()
plt.show()