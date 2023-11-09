import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import seaborn as sb
from scipy.optimize import curve_fit
import pandas as pd

df = pd.read_csv("zad4_lista1.txt", delimiter="  ", header=None)
x, y = df[0], df[1]


def mnk(x, y): #metoda najmniejszych kwadratow
    b1 = (sum((x - np.mean(x)) * y) / sum((x - np.mean(x))**2))
    b0 = np.mean(y) - b1 * np.mean(x)
    return b1, b0

b1, b0 = mnk(x,y)
print(f"b1 i b0 ze wzoru: {b1}, {b0}")

n1 = np.polyfit(x, y, 1)
print(f"b1 i b0 z polyfit: {n1}")

x1 = np.linspace(-15, 15)
y1 = b0 + b1*x1

plt.figure(figsize=(16,8))
plt.scatter(x, y, label = "dane rozproszone")
plt.plot(x1, y1, "r--", label = "prosta z wyestymowanymi współczynnikami b0 i b1")
plt.title(f"Regresja liniowa y={b0}+{b1}*x")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()



y_est = []
for i in range(len(x)):
  y_est.append(b0 + b1*x[i])

e_i = []
for j in range(len(y)):
  e_i.append(y[j] - y_est[j])

plt.boxplot(e_i)
plt.axhline(np.mean(e_i), label="średnia")
plt.legend()
plt.title("Boxplot residuów")



def R2(y, y_est):
    r2 = (np.sum((y_est - np.mean(y))**2)) / (np.sum((y - np.mean(y))**2))
    return r2

R_2 = R2(y, y_est)
print(f"Współczynnik dererminacji wynosi {R_2}")



q1 = np.quantile(e_i, 0.25)
q3 = np.quantile(e_i, 0.75)
IQR = q3-q1

a1 = q1 - 1.5*IQR
a2 = q3 + 1.5*IQR

e_i_ucieta = []
X_ucieta = []
Y_ucieta = []


for m in range(len(e_i)):
    if a1<=e_i[m]<=a2:
      e_i_ucieta.append(e_i[m])
      X_ucieta.append(x[m])
      Y_ucieta.append(y[m])

plt.boxplot(e_i_ucieta)
plt.title("Boxplot uciętych residuów")
plt.axhline(np.mean(e_i_ucieta), label="średnia")
plt.legend()



b1_u, b0_u = np.polyfit(X_ucieta, Y_ucieta, 1)

x1 = np.linspace(-15, 15)
y1 = b0_u + b1_u*x1

plt.figure(figsize=(16,8))
plt.scatter(X_ucieta, Y_ucieta, label = "dane rozproszone")
plt.plot(x1, y1, "r--", label = "prosta z wyestymowanymi współczynnikami b0 i b1")
plt.title(f"Regresja liniowa y={b0_u}+{b1_u}*x")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()



y_est_ucieta = []
for i in range(len(X_ucieta)):
  y_est_ucieta.append(b0_u + b1_u*X_ucieta[i])

R_2_u = R2(Y_ucieta, y_est_ucieta)
print(f"Współczynnik dererminacji wynosi {R_2_u}")