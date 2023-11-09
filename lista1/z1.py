import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import pandas as pd

#x - zmienna objasniajaca
#y - zmienna objasniana

df = pd.read_csv("zad1_lista1.txt", delimiter="  ", header=None)
size = len(df)

x, y = df[0], df[1]


def polynomial(x, a, b, c):
    y1 = a*x**2 + b*x + c
    return y1

"""
a, b, c = np.polyfit(x, y, 2)
print(f"a = {a}, b = {b}, c = {c}")

x2 = np.linspace(min(x), max(x), 1000)
y2 = polynomial(x2, a, b, c)"""


a, b, c = curve_fit(polynomial, x, y)[0]
print(f"a = {a}, b = {b}, c = {c}")

x2 = np.linspace(min(x), max(x), 1000)
y2 = polynomial(x2, a, b, c)

plt.plot(x2,y2)
plt.scatter(x,y, s=4)
plt.xlabel("x")
plt.ylabel("y")
plt.title(f"Scattered data and polynomial")
plt.show()