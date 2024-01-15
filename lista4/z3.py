import numpy as np
import matplotlib.pyplot as plt


def func(sigm, n):
    gamm = np.zeros(41)
    gamm[20] = sigm**2  
    gamm_est = []

    X = np.random.normal(0, sigm, n+1)
    
    for h in range(-20, 21):
        dosumy = [] 
        
        for t in range(1, n - abs(h)):
            dosumy.append((X[t] - np.mean(X)) * (X[t + abs(h)] - np.mean(X)))

        gamm_est.append(1/n * sum(dosumy))

    RMSE = np.sqrt(np.mean((gamm - gamm_est)**2))
    MAE = np.sqrt(np.mean(abs(gamm - gamm_est)))

    return gamm_est, gamm, RMSE, MAE


n = 1000
sigm = np.sqrt(2)

gamm_est, gamm, RMSE, MAE = func(sigm, n)

plt.stem(np.arange(-20, 21), gamm_est, "r", label="gamm_est")
plt.stem(np.arange(-20, 21), gamm, "b", label="gamm")
plt.legend()
plt.show()

plt.boxplot([gamm_est, gamm], labels=['gamm_est', 'gamm'])
plt.show()

print(RMSE, MAE)