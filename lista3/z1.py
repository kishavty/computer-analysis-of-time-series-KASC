import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

M = 1000
b_0 = 3
b_1 = 2


def func(n, sigm, alpha, b0 = 3, b1 = 2):
    """
        Perform Monte Carlo simulations to estimate confidence intervals for regression coefficients and intercepts.

    """
    count1 = 0
    count2 = 0
    count3 = 0
    count4 = 0

    x = np.arange(1, n+1, 1)

    for i in range(M):
        epsilon = np.random.normal(0, sigm**2, n)
        y = np.zeros(n)
        y = b_0 + b_1*x + epsilon
        estb1 = sum((x - np.mean(x)) * y) / sum((x - np.mean(x))**2)
        estb0 = np.mean(y) - estb1 * np.mean(x)
        esty = estb0 + estb1*x
        s = np.sqrt(sum((y - esty)**2) / (n - 2))

        do_z1_a = np.random.normal(0,1,M)
        z1_a = np.quantile(do_z1_a, 1-alpha/2)

        do_t1_a = np.random.standard_t(n-2, M)
        t1_a = np.quantile(do_t1_a, 1-alpha/2)

        a1 = estb1 - z1_a * (sigm / (np.sqrt(sum((x - np.mean(x))**2))))
        b1 = estb1 + z1_a * (sigm / (np.sqrt(sum((x - np.mean(x))**2))))

        a2 = estb1 - t1_a * s / np.sqrt(sum((x - np.mean(x))**2))
        b2 = estb1 + t1_a * s / np.sqrt(sum((x - np.mean(x))**2))

        a3 = estb0 - z1_a * sigm * np.sqrt( 1/n + np.mean(x)**2 / sum((x - np.mean(x))**2))
        b3 = estb0 + z1_a * sigm * np.sqrt( 1/n + np.mean(x)**2  /sum((x - np.mean(x))**2))
            
        a4 = estb0 - t1_a * s * np.sqrt( 1/n + np.mean(x)**2 / sum((x - np.mean(x))**2))
        b4 = estb0 + t1_a * s * np.sqrt( 1/n + np.mean(x)**2 / sum((x - np.mean(x))**2))

        if a1 < b_1 < b1:
            count1 += 1

        if a2 < b_1 < b2:
            count2 += 1

        if a3 < b_0 < b3:
            count3 += 1

        if a4 < b_0 < b4:
            count4 += 1

    return count1/M, (count2)/M, (count3)/M, (count4)/M


####
n = 100
sigm = 1
alphas = [0.01, 0.05, 0.1]

for alpha in alphas:
    print(n, sigm, alpha)
    print(func(n, sigm, alpha))

###
n = 100
alpha = 0.05
sigmas = [0.01, 0.5, 1] 

for sigm in sigmas:
    print(n, sigm, alpha)
    print(func(n, sigm, alpha))


###
alpha = 0.05
sigm = 1
ns = [100, 200, 300]

for n in ns:
    print(n, sigm, alpha)
    print(func(n, sigm, alpha))
