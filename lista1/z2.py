import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import pandas as pd

#podstawa 11 p=5, 25 p=12, (2p+1)

df = pd.read_csv("zad2_lista1.txt", delimiter="  ", header=None)
x = df[0]
size = len(x)


def srednia_ruchoma(size, p):
    m = 2*p+1
    srednia = []
    for k in range(p+1,size-p):
        l_xsr_ruch=[]
        for j in range(-p,p):
            n = x[k+j]
            l_xsr_ruch.append(n)
        ruchoma = 1/m *sum(l_xsr_ruch)
        srednia.append(ruchoma)
    return srednia
"""

for p in (5, 12, 50):
    plt.plot(x)
    plt.plot(srednia_ruchoma(size, p), label=f"p={p}")
    plt.title(f"Średnia ruchoma, p = {5,12,50}")
    plt.legend()
plt.show()"""

plt.plot(srednia_ruchoma(size, 5), label=f"p={5}")
plt.title(f"Średnia ruchoma, p = {5,12,50}")
plt.legend()
plt.show()