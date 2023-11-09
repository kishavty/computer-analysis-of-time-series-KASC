import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import pandas as pd

df_x = pd.read_csv("zad2_lista1.txt", delimiter="  ", header=None)
x = df_x[0]

df_y = pd.read_csv("zad3_lista1.txt", delimiter="  ", header=None)
y = df_y[0]

size = len(x)

def mnk(x, y): #metoda najmniejszych kwadratow
    b1 = (sum((x - np.mean(x)) * y) / sum((x - np.mean(x))**2))
    b0 = np.mean(y) - b1 * np.mean(x)
    return b1, b0

def polynomial(x, a, b):
    y1 = a*x + b
    return y1


###wykres

plt.scatter(x, y, s=5, label="scatter plot danych")

b1, b0 = mnk(x, y)
a, b = curve_fit(polynomial, x, y)[0]

x1 = np.linspace(-1, 10, 100)
y1 = b1*x1 + b0

y2 = a*x1 + b

plt.plot(x1, y1, "b", alpha = 0.8, label="regr. lin. z estymowanych współczynników")
plt.plot(x1, y2, "r", linestyle = ':', alpha = 0.8, label="regr. lin. z curve_fit")
plt.title(f"dane + dopasowana funkcja liniowa y = b_1*x+b_0, gdzie b_1={round(b1, 4)}, b_0={round(b0, 4)}")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()


#############

def sr_ruch(size, p):
    m = 2 * p + 1
    x_srednia = []
    y_srednia = y[12:size-12]
    for k in range(p, size - p):
        lista_xsr_ruch = []
        for j in range(-p, p + 1):
            size = x[k + j]
            lista_xsr_ruch.append(size)
        ruchoma = 1 / m * sum(lista_xsr_ruch)
        x_srednia.append(ruchoma)
    return x_srednia, y_srednia


x_srednia, y_srednia = sr_ruch(size, 12)

a_2, b_2 = np.polyfit(x_srednia,y_srednia,1)

x1 = np.linspace(min(x_srednia),max(x_srednia),100)
y1 = a_2*x1 + b_2
plt.plot(x1, y1, "b", linewidth = 4, alpha = 0.8, label="regr. lin. z polyfit")


y2 = b1*x1 + b0
plt.plot(x1, y2, "r", linewidth = 4, linestyle=":", alpha = 0.8, label="regr. lin. z estymowanych współczynników")

plt.scatter(x_srednia, y_srednia, s=5, label="scatter plot danych")
plt.title(f"dane rozproszone wygładzone + dopasowana funkcja liniowa o wspołczynnikach : b_1={round(b1, 4)}, b_0={round(b0, 4)}")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()



