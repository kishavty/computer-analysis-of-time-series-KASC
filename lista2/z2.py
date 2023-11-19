import numpy as np
import matplotlib.pyplot as plt

# e)
M = 1000
beta_1  = 3

def func(n, sigm):
    x = np.linspace(1, 100, n)
    est_b = np.zeros(M)
    for i in range(M):
        y = np.zeros(n)
        epsilon = np.random.normal(0, sigm, n)
        y = beta_1 * x + epsilon
        est_b[i] = sum(x * y) / sum(x**2)
    emp_mean_est_b = np.mean(est_b)
    emp_var_est_b = np.var(est_b)
    return est_b, emp_mean_est_b, emp_var_est_b


# n = 100
sigmas  = [10, 20, 30]

for sigm in sigmas:
    est_b, emp_mean_est_b, emp_var_est_b = func(100, sigm)

    theor_mean_est_b = beta_1
    theor_var_est_b = (sigm**2) / sum(np.linspace(1, 100, 100)**2)

    print(f"n = 100, sigm = {sigm}")
    print(f"Theoretical mean: {theor_mean_est_b}, empirical mean: {emp_mean_est_b}")
    print(f"Theoretical var: {theor_var_est_b}, empirical var: {emp_var_est_b}")

    plt.boxplot(est_b)
    plt.title(f"Empirical boxplot, n = {100}, sigm = {sigm}")
    plt.show()


# sigm = 20
ns  = [100, 500, 1000]

for n in ns:
    est_b, emp_mean_est_b, emp_var_est_b = func(n, 20)

    theor_mean_est_b = beta_1
    theor_var_est_b = (20**2) / sum(np.linspace(1, 100, n)**2)

    print(f"n = {n}, sigm = 20")
    print(f"Theoretical mean: {theor_mean_est_b}, empirical mean: {emp_mean_est_b}")
    print(f"Theoretical var: {theor_var_est_b}, empirical var: {emp_var_est_b}")

    plt.boxplot(est_b)
    plt.title(f"Empirical boxplot, n = {n}, sigm = 20")
    plt.show()





