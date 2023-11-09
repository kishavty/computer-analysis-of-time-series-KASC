import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import pandas as pd


df = pd.read_csv("zad6_lista1.txt", delimiter="  ", header=None)
x, y = df[0], df[1]
size = len(x)

t_y = [] #transformed y
for i in y:
    t_y.append(np.log(i))


def mnk(x, y): #metoda najmniejszych kwadratow
    b1 = (sum((x - np.mean(x)) * y) / sum((x - np.mean(x))**2))
    b0 = np.mean(y) - b1 * np.mean(x)
    return b1, b0


a, b = np.polyfit(x, t_y, 1)
print(f"a = {a}, b = {b}")


b1, b0 = mnk(x, t_y)
y1 = b1*x + b0

y2 = a*x + b

plt.figure(figsize=(10,8))
plt.scatter(x, t_y, s=4)
plt.plot(x, y1, "b", alpha = 0.8, label="regr. lin. z estymowanych współczynników")
plt.plot(x, y2, "r", linestyle = ':', alpha = 0.8, label="regr. lin. z curve_fit")
plt.title(f"dane + dopasowana funkcja liniowa y = b_1*x+b_0, gdzie b_1={round(b1, 4)}, b_0={round(b0, 4)}")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()

#wspolczynnik determinancji
r2 = np.sum([(i - np.mean(t_y))**2 for i in y1]) / np.sum([(i - np.mean(t_y))**2 for i in t_y])
print(r2) 


plt.figure(figsize=(10,8))
plt.scatter(x, y, s=4, label="dane poczatkowe")
plt.plot(sorted(x), sorted([np.exp(b0) * np.exp(b1 * i) for i in x]), "r", label="y=a*exp(bx)")
plt.title(f"dane poczatkowe + dobrana f. y = a*exp(bx), gdzie a=exp(b0)={round(np.exp(b0), 4)}, b=b1={round(b1, 4)}")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()


r2_2 = np.sum([(i - np.mean(y))**2 for i in [np.exp(b0) * np.exp(b1 * x) ]]) / np.sum([(i - np.mean(y))**2 for i in y])
print(r2_2) 