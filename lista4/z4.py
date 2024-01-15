import numpy as np
import matplotlib.pyplot as plt


def autocorrelation_emp(sigm, n, theta):
    Z = np.random.normal(0, sigm**2, n)
    X = np.empty(n)
    ro_est = []
    
    for t in range(1, n):
        X[t] = Z[t] + theta * Z[t - 1]
    
    for h in range(-10, 11):
        dosumy = np.empty(n - abs(h))
        for t in range(0, n - abs(h)):
            dosumy[t] = (X[t] - np.mean(X)) * (X[t + abs(h)] - np.mean(X))
        
        gamma = np.sum(dosumy) / (n - abs(h))
        
        dosumy_2 = (X - np.mean(X))**2
        
        ro_est.append(gamma / (np.sum(dosumy_2) / n))
        
    return ro_est

def autocorrelation_teo(theta):
    cor_teo = np.zeros(len(np.arange(-10, 11)))
    for i, h in enumerate(np.arange(-10, 11)):
        if h == 0:
            cor_teo[i] = 1
        elif abs(h) == 1:
            cor_teo[i] = theta / (1 + theta**2)
    return cor_teo

sigm = np.sqrt(2)
n = 1000
theta = 0.5

result_emp = autocorrelation_emp(sigm, n, theta)
result_teo = autocorrelation_teo(theta)


plt.scatter(np.arange(-10, 11), result_emp, label = "empirical")
plt.scatter(np.arange(-10, 11), result_teo, label = "theoretical")
plt.title('Autokorelacja')
plt.grid(True)
plt.show()