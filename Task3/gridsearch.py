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

def marglll(y, phi, mu, A, k, alpha, std):
    first = 0.5 * (k+1) * np.log(alpha) - 0.5 * len(y) * np.log(std ** 2)
    second = - (0.5 / (std ** 2)) * (np.linalg.norm(y - phi @ mu, ord=2) ** 2)
    third = (alpha / 2) * mu.T @ mu - 0.5 * np.log(np.linalg.det(A)) - 0.5 * len(y) * np.log(2 * np.pi)
    return first + second + third


def compute_rmse(y, y_pred):
    return (np.mean((y - y_pred) ** 2)) ** 0.5


def main():
    # data
    traindata = np.loadtxt("lin_reg_train.txt", dtype=float)
    testdata = np.loadtxt("lin_reg_test.txt", dtype=float)

    x_train = traindata[:, 0]
    y_train = traindata[:, 1]
    idx_train = np.argsort(x_train)
    x_train = x_train[idx_train]
    y_train = y_train[idx_train]
    print(y_train.shape)


    x_test = testdata[:, 0]
    y_test = testdata[:, 1]
    idx_test = np.argsort(x_test)
    x_test = x_test[idx_test]
    y_test = y_test[idx_test]

    b = [1, 10, 100]
    k = 20
    std = 0.1
    beta = 1 / (std ** 2)
    alpha = 0.01

    rmse_train = np.zeros((1, len(b)))
    rmse_test = np.zeros((1, len(b)))

    avg_lll_train = np.zeros((1, len(b)))
    avg_lll_test = np.zeros((1, len(b)))

    marg_lll_train = np.zeros((1, len(b)))


    for i, bi in enumerate(b):
        phitrain = sefeature_map(x_train, k, bi)
        phitest = sefeature_map(x_test, k, bi)
        phi_train = np.c_[np.ones(len(x_train)), phitrain]
        phi_test = np.c_[np.ones(len(x_test)), phitest]

        n = phi_train.shape[1]
        S0 = (1 / alpha) * np.identity(n)
        print(n)

        # posterior
        invSn = np.linalg.inv(S0) + beta * phi_train.T @ phi_train
        Sn = np.linalg.inv(invSn)
        mun = beta * Sn @ phi_train.T @ y_train[:, np.newaxis] # 20x1

        # predictive
        mu_pred_test = mun.T @ phi_test.T
        mu_pred_train = mun.T @ phi_train.T
        mu_pred_test = np.squeeze(mu_pred_test)
        on_pred_test = (1 / beta) + phi_test @ Sn @ phi_test.T
        on_pred_train = (1 / beta) + phi_test @ Sn @ phi_test.T
        on_pred_diag = np.diag(on_pred_test)

        marg_lll_train[:, i] = marglll(y_train, phi_train, mun, invSn, k, alpha, std)



        # rmse
        rmse_test[:, i] = compute_rmse(y_test, mu_pred_test)
        rmse_train[:, i] = compute_rmse(y_train, mu_pred_train)


        # average loglikelihood


        print(f'RMSE Test for beta={bi}:', rmse_test[:, i])
        print(f'RMSE Train for beta={bi}:', rmse_train[:, i])
        print(f'average Loglikelihood test set for beta={bi}:', avg_lll_test[:, i])
        print(f'average Loglikelihood train set for beta={bi}:', avg_lll_train[:, i])
        print(f'Marginal Loglikelihood test set for beta={bi}:', marg_lll_test[:, i])
        print(f'Marginal Loglikelihood train set for beta={bi}:', marg_lll_train[:, i])

        # plot
        plt.figure()
        plt.scatter(x_train, y_train, c='black')
        plt.plot(x_test, mu_pred_test.T)
        plt.fill_between(x_test, mu_pred_test.T - np.sqrt(on_pred_diag), mu_pred_test.T + np.sqrt(on_pred_diag),
                         color='blue', alpha=0.3)
        plt.fill_between(x_test, mu_pred_test.T - np.sqrt(on_pred_diag) * 2, mu_pred_test.T + np.sqrt(on_pred_diag) * 2,
                         color='blue', alpha=0.2)
        plt.fill_between(x_test, mu_pred_test.T - np.sqrt(on_pred_diag) * 3, mu_pred_test.T + np.sqrt(on_pred_diag) * 3,
                         color='blue', alpha=0.1)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'SE Feature Regression with beta={bi}')
        plt.show()






if __name__ == '__main__':
    main()
