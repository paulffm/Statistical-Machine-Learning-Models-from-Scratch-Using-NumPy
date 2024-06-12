import numpy as np
from matplotlib import pyplot as plt

def avgloglike(y, mu, cov, d):
    y = y[:, np.newaxis]
    y_m = (y - mu.T)

    first = - 0.5 * d * np.log(2 * np.pi) - 0.5 * np.log(np.linalg.det(cov))
    second = -0.5 * y_m.T @ (np.linalg.inv(cov)) @ y_m
    return (first + second) / len(y)

def compute_rmse(y, y_pred):
    return (np.mean((y - y_pred) ** 2)) ** 0.5

def N(x, mean, cov):
    x = x[:, np.newaxis]
    d = len(mean)
    e = np.exp(-0.5*np.matmul((x-mean).transpose(), np.matmul(np.linalg.inv(cov), x-mean)))
    #np.linalg.det(cov)
    return e/(((2*np.pi)**(d*0.5))*np.sqrt(1e-5))


def main():
    # data
    traindata = np.loadtxt("lin_reg_train.txt", dtype=float)
    testdata = np.loadtxt("lin_reg_test.txt", dtype=float)

    x_train = traindata[:, 0]
    y_train = traindata[:, 1]
    idx_train = np.argsort(x_train)
    x_train = x_train[idx_train]
    y_train = y_train[idx_train]


    x_test = testdata[:, 0]
    y_test = testdata[:, 1]
    idx_test = np.argsort(x_test)
    x_test = x_test[idx_test]
    y_test = y_test[idx_test]



    Xp_train = np.c_[np.ones(len(x_train)), x_train]
    Xp_test = np.c_[np.ones(len(x_test)), x_test]

    n = Xp_train.shape[1]
    std = 0.1
    beta = 1 / (std ** 2)
    alpha = 0.01
    S0 = (1 / alpha) * np.identity(n)


    # wmap = np.linalg.inv(Xp.T @ Xp + (alpha / beta) * np.identity(n)) @ Xp.T @ y_train #2x1

    # posterior
    invSn = S0 + beta * Xp_train.T @ Xp_train
    Sn = np.linalg.inv(invSn)
    mun = beta * Sn @ Xp_train.T @ y_train[:, np.newaxis]


    # predictive
    mu_pred_test = mun.T @ Xp_test.T
    mu_pred_train = mun.T @ Xp_train.T
    mu_pred_test = np.squeeze(mu_pred_test)
    on_pred_test = (1 / beta) + Xp_test @ Sn @ Xp_test.T
    print(np.linalg.det(on_pred_test))
    on_pred_train = (1 / beta) + Xp_train @ Sn @ Xp_train.T
    on_pred_diag= np.diag(on_pred_test)

    # average loglikelihood
    d = 2 #Xp_test.shape[1]

    # avg_lll_test = avgloglike(y_test, mu_pred_test, on_pred_test, d)
    # avg_lll_train = avgloglike(y_train, mu_pred_train, on_pred_train, d)
    print(N(y_test, mu_pred_test, on_pred_test))
    avg_lll_test = np.log(N(y_test, mu_pred_test, on_pred_test)) / len(y_test)
    avg_lll_train = np.log(N(y_train, mu_pred_train, on_pred_train)) / len(y_train)
    print('average Loglikelihood for test set', avg_lll_test)
    print('average Loglikelihood for train set', avg_lll_train)



    # rmse
    rmse_test = compute_rmse(y_test, mu_pred_test)
    rmse_train = compute_rmse(y_train, mu_pred_train)
    print('RMSE Test', rmse_test)
    print('RMSE Train', rmse_train)

    # plot
    plt.figure()
    plt.scatter(x_train, y_train, c='black')
    plt.plot(x_test, mu_pred_test.T)
    plt.fill_between(x_test, mu_pred_test.T - np.sqrt(on_pred_diag), mu_pred_test.T + np.sqrt(on_pred_diag),
                     color='blue', alpha=0.3)
    plt.fill_between(x_test, mu_pred_test.T - np.sqrt(on_pred_diag)*2, mu_pred_test.T + np.sqrt(on_pred_diag)*2,
                     color='blue', alpha=0.2)
    plt.fill_between(x_test, mu_pred_test.T - np.sqrt(on_pred_diag)*3, mu_pred_test.T + np.sqrt(on_pred_diag)*3,
                     color='blue', alpha=0.1)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Bayesian Linear Regression')
    plt.show()


if __name__ == '__main__':
    main()
