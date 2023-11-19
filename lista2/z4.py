import numpy as np 
import matplotlib.pyplot as plt

import scipy.stats as st
M = 1000


def func(n):
    x = np.linspace(0, 10, n)
    student_est_b0 = np.zeros(M)
    student_est_b1 = np.zeros(M)
    for i in range(M):
        epsilon = np.random.normal(0, sigm**2, n)
        y = np.zeros(n)
        y = b0 + b1*x + epsilon
        est_b1 = sum((x - np.mean(x)) * y) / sum((x - np.mean(x))**2)
        est_b0 = np.mean(y) - est_b1 * np.mean(x)
        s = np.sqrt(sum((y - np.mean(y))**2) / (n - 2))

        student_est_b0[i] = (est_b0 - b0) / (sigm * np.sqrt(1/n + (np.mean(x)**2 / sum((x - np.mean(x))**2))))
        student_est_b1[i] = (est_b1 - b1) / (sigm / np.sqrt(sum((x - np.mean(x))**2)))

    return np.sort(student_est_b0), np.sort(student_est_b1)

n = 1000
sigm = 1
b0 = 3
b1 = 2

sample = func(n)

plt.plot( sample[0],  np.arange(1, len(sample[0]) + 1) / len(sample[0]), label = "empirical distribution")
plt.plot(st.t.ppf(np.linspace(0, 10, n), n-2 ), np.linspace(0, 10, n), alpha = 0.7, label = "theoretical distribution")
plt.title("Empirical and theoretical distribution plot for b0")
plt.show()


plt.plot( sample[1],  np.arange(1, len(sample[1]) + 1) / len(sample[1]), label = "empirical distribution")
plt.plot(st.t.ppf(np.linspace(0, 10, n), n-2 ), np.linspace(0, 10, n), alpha = 0.7, label = "theoretical distribution")
plt.title("Empirical and theoretical distribution plot for b1")
plt.show()