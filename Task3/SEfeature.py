import numpy as np
from matplotlib import pyplot as plt


'''https://towardsdatascience.com/bayesian-linear-regression-in-python-using-machine-learning-to-predict-student-grades
-part-2-b72059a8ac7e'''

'''. In the linear regression method, the model parameters w are calculated by finding the parameters
 which minimize the sum of squared errors on the training data. In contrast, Bayesian Linear Regression assumes the 
 responses are sampled from a probability distribution such as the normal (Gaussian) distribution: y-N(B.T X, o^2)
'''

'''In Bayesian Models, not only is the response assumed to be sampled from a distribution, but so are the parameters. 
The objective is to determine the posterior probability distribution for the model parameters given the inputs, X, and 
outputs, y:'''
def sefeature_map(X, k, beta):
    phi = np.zeros(((X.shape[0], k)))
    for i in range(1, k+1):
        phi[:, i-1] = np.exp(- 0.5 * beta * ((X - (i * 0.1 - 1)) ** 2))
    return phi


def N(x, mean, cov):
    x = x[:, np.newaxis]
    d = len(mean)
    e = np.exp(-0.5*np.matmul((x-mean).transpose(), np.matmul(np.linalg.inv(cov), x-mean)))
    #np.linalg.det(cov)
    return e/(((2*np.pi)**(d*0.5))*np.sqrt(1e-5))

def compute_rmse(y, y_pred):
    return (np.mean((y - y_pred) ** 2)) ** 0.5

def multivariategaussian(X, mean, cov):
    d = mean.shape[1]
    x_m = X-mean
    e = np.exp(-0.5 * x_m @ np.linalg.inv(cov) @ x_m.T)
    return e / (((2 * np.pi) ** (d * 0.5)) * np.sqrt(np.linalg.det(cov)))

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

    b = 10
    k = 20

    phitrain = sefeature_map(x_train, k, b)
    phitest = sefeature_map(x_test, k, b)
    phi_train = np.c_[np.ones(len(x_train)), phitrain]
    phi_test = np.c_[np.ones(len(x_test)), phitest]

    n = phi_train.shape[1]
    std = 0.1
    beta = 1 / (std ** 2)
    alpha = 0.01
    S0 = (1 / alpha) * np.identity(n)

    # posterior
    invSn = np.linalg.inv(S0) + beta * phi_train.T @ phi_train
    Sn = np.linalg.inv(invSn)
    print(Sn.shape) #20x20
    mun = beta * Sn @ phi_train.T @ y_train[:, np.newaxis] # 20x1
    print(mun.shape)

    # predictive
    mu_pred_test = mun.T @ phi_test.T
    print(mu_pred_test)
    mu_pred_train = mun.T @ phi_train.T
    mu_pred_test = np.squeeze(mu_pred_test)
    on_pred_test = (1 / beta) + phi_test @ Sn @ phi_test.T # 50x50
    on_pred_train = (1 / beta) + phi_train @ Sn @ phi_train.T  # 50x50
    on_pred_diag = np.diag(on_pred_test)

    # rmse
    rmse_test = compute_rmse(y_test, mu_pred_test)
    rmse_train = compute_rmse(y_train, mu_pred_train)
    print('RMSE Test', rmse_test)
    print('RMSE Train', rmse_train)

    # average loglikelihood
    avg_lll_test = np.log(N(y_test, mu_pred_test, on_pred_test)) / 100
    avg_lll_train = np.log(N(y_train, mu_pred_train, on_pred_train)) / 50
    print('average Loglikelihood for test set', avg_lll_test)
    print('average Loglikelihood for train set', avg_lll_train)

    # plot
    plt.figure()
    plt.scatter(x_train, y_train, c='black')
    plt.fill_between(x_test, mu_pred_test.T - np.sqrt(on_pred_diag), mu_pred_test.T + np.sqrt(on_pred_diag),
                     color='blue', alpha=0.3)
    plt.fill_between(x_test, mu_pred_test.T - np.sqrt(on_pred_diag) * 2, mu_pred_test.T + np.sqrt(on_pred_diag) * 2,
                     color='blue', alpha=0.2)
    plt.fill_between(x_test, mu_pred_test.T - np.sqrt(on_pred_diag) * 3, mu_pred_test.T + np.sqrt(on_pred_diag) * 3,
                     color='blue', alpha=0.1)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('SE Feature Regression')
    plt.show()






    '''Xp_train = np.c_[np.ones(len(x_train)), x_train]
    Xp_test = np.c_[np.ones(len(x_test)), x_test]

    n = Xp_train.shape[1]
    std = 0.1
    beta = 1 / (std ** 2)
    alpha = 0.01
    S0 = (1 / alpha) * np.identity(n)


    # wmap = np.linalg.inv(Xp.T @ Xp + (alpha / beta) * np.identity(n)) @ Xp.T @ y_train #2x1

    # posterior
    invSn = np.linalg.inv(S0) + beta * Xp_train.T @ Xp_train
    Sn = np.linalg.inv(invSn)
    mun = beta * Sn @ Xp_train.T @ y_train[:, np.newaxis]


    # predictive
    mu_pred_test = mun.T @ Xp_test.T
    mu_pred_train = mun.T @ Xp_train.T
    mu_pred_test = np.squeeze(mu_pred_test)
    on_pred = (1 / beta) + Xp_test @ Sn @ Xp_test.T
    on_pred_diag= np.diag(on_pred)


    # rmse
    rmse_test = compute_rmse(y_test, mu_pred_test)
    rmse_train = compute_rmse(y_train, mu_pred_train)
    print('RMSE Test', rmse_test)
    print('RMSE Train', rmse_train)


    # plot
    plt.figure()
    plt.scatter(x_train, y_train, c='black')
    plt.plot(x_test, mu_pred_test.T)
    plt.fill_between(x_test, mu_pred_test.T - np.sqrt(on_pred_diag[0]), mu_pred_test.T + np.sqrt(on_pred_diag[0]))
    plt.fill_between(x_test, mu_pred_test.T - np.sqrt(on_pred_diag[1]), mu_pred_test.T + np.sqrt(on_pred_diag[1]))
    plt.fill_between(x_test, mu_pred_test.T - np.sqrt(on_pred_diag[2]), mu_pred_test.T + np.sqrt(on_pred_diag[2]))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Bayesian Linear Regression')
    plt.show()'''




if __name__ == '__main__':
    main()
