import numpy as np
from matplotlib import pyplot as plt

def multivariategaussian(X, mean, cov):
    d = mean.shape[1]
    x_m = X-mean
    e = np.exp(-0.5 * x_m @ np.linalg.inv(cov) @ x_m.T)
    return e / (((2 * np.pi) ** (d * 0.5)) * np.sqrt(np.linalg.det(cov)))

def main():
    # data
    traindata = np.loadtxt("lin_reg_train.txt", dtype=float)
    print(traindata.shape)
    testdata = np.loadtxt("lin_reg_test.txt", dtype=float)

    x_train = traindata[:, 0]
    y_train = traindata[:, 1]
    idx_train = np.argsort(x_train)
    x_train = x_train[idx_train]
    y_train = y_train[idx_train]
    print(x_train)

    x_test = testdata[:, 0]
    y_test = testdata[:, 1]
    idx_test = np.argsort(x_test)
    x_test = x_test[idx_test]
    y_test = y_test[idx_test]


    # posterior: N x 2
    Xp = np.c_[np.ones(len(x_train)), x_train]
    n = Xp.shape[1]
    std = 0.1
    beta = 1 / (std ** 2)
    alpha = 0.01

    S0 = (1 / alpha) * np.identity(n)
    print(S0)
    mu0 = np.zeros((1, 2))

    # prior
    p_w = multivariategaussian(Xp, mu0, S0)
    print(p_w)
    # p_w = multivariategaussian(X, mu0, S0)



    # update
    invSn = np.linalg.inv(S0) + beta * Xp.T @ Xp
    # Sn = np.linalg.inv(invSn)
    mun = beta * Sn @ Xp.T @ y_train

    # derive w:
    w = np.linalg.inv(Xp.T @ Xp + (alpha / beta) * np.identity(2)) @ Xp.T @ y_train
    # w = multivariategaussian(Xp, mu0, S0)



    

    # predictive
    varn = (1 / beta) + Xp.T @ Sn @ Xp


if __name__ == '__main__':
    main()
