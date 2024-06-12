import numpy as np
from matplotlib import pyplot as plt

''' λ is the ridge coefficient that decides how much we want to penalize the flexibility of our model.  this technique 
discourages learning a more complex or flexible model, so as to avoid the risk of overfitting.'''
'https://towardsdatascience.com/regularization-in-machine-learning-76441ddcf99a'

'''regressor phi(x)= [1 x x^2 x^3 ... x^d] is polynomial but the model is still linear y(x) = w.T phi(x), therefore the 
learning methods for linear regression can still be applied for a regression with polynomials.'''

def linridgeregression(X, k, y):
    n = len(X)
    Xp = np.c_[np.ones(n), X]
    return np.linalg.inv(Xp.T @ Xp + k * np.identity(Xp.shape[1])) @ Xp.T @ y

def polyridgeregression(Xp, k, y):
    return np.linalg.inv(Xp.T @ Xp + k * k * np.identity(Xp.shape[1])) @ Xp.T @ y


def map_polynomial(X, degree):
    x_poly = []
    # For each datapoint
    for xi in X.reshape(-1):
        xi_poly = [xi ** d for d in range(degree + 1)]
        x_poly.append(xi_poly)
    return np.array(x_poly)


def compute_linpredictions(beta, X):
    n = len(X)
    Xp = np.c_[np.ones(n), X]

    return Xp @ beta


def compute_polypredictions(beta, Xp):
    return Xp @ beta


def compute_rmse(y, y_pred):
    return (np.mean((y - y_pred) ** 2)) ** 0.5


def polynomial_regression(x_train, x_test, y_train, y_test, d, k):
    x_train_poly = map_polynomial(x_train, d)
    x_test_poly = map_polynomial(x_test, d)
    print(x_train_poly.shape)
    beta_train_poly = polyridgeregression(x_train_poly, k, y_train)

    y_pred_train = compute_polypredictions(beta_train_poly, x_train_poly)
    y_pred_test = compute_polypredictions(beta_train_poly, x_test_poly)
    return y_pred_train, y_pred_test


def plot_regression(x_train, y_train, x_test, y_pred, k):

    plt.figure()
    plt.scatter(x_train, y_train, c='black')
    plt.plot(x_test, y_pred, color='blue')
    plt.title(f'Ridge Regression with k = {k}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

def plot_polyregression(X, y, y_pred, d):
    plt.figure()
    plt.scatter(X, y, c='black')
    plt.plot(X, y_pred, color='blue')
    plt.title(f'Polynomial Ridge Regression with degree = {d}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


def main():

    # load data
    traindata = np.loadtxt("lin_reg_train.txt", dtype=float)
    testdata = np.loadtxt("lin_reg_test.txt", dtype=float)

    x_train = traindata[:, 0]
    y_train = traindata[:, 1]
    index_train = np.argsort(x_train)
    x_train = x_train[index_train]
    y_train = y_train[index_train]

    x_test = testdata[:, 0]
    y_test = testdata[:, 1]
    index_test = np.argsort(x_test)
    x_test = x_test[index_test]
    y_test = y_test[index_test]

    k = 0.01

    ### RIDGE REGRESSION ###
    # perform ridge regression
    beta_train = linridgeregression(x_train, k, y_train)

    # make predictions
    ylin_pred_train = compute_linpredictions(beta_train, x_train)
    ylin_pred_test = compute_linpredictions(beta_train, x_test)

    # compute rmse
    rmselin_train = compute_rmse(y_train, ylin_pred_train)
    rmselin_test = compute_rmse(y_test, ylin_pred_test)
    print('Root Mean Squared Train error:', rmselin_train)
    print('Root Mean Squared Test error:', rmselin_test)

    plot_regression(x_train, y_train, x_test, ylin_pred_test, k)



    # Polynomial:
    d = [2, 3, 4]
    ypoly_train = np.zeros((len(d), len(x_train)))
    ypoly_test = np.zeros((len(d), len(x_test)))

    for i, di in enumerate(d):
        ypoly_pred_train, ypoly_pred_test = polynomial_regression(x_train, x_test, y_train, y_test,
                                                                  di, k)
        rmsepoly_train = compute_rmse(y_train, ypoly_pred_train)
        rmsepoly_test = compute_rmse(y_test, ypoly_pred_test)
        print(f'RMSE Train with degree: {di}', rmsepoly_train)
        print(f'RMSE Test with degree: {di}', rmsepoly_test)

        ypoly_train[i, :] = ypoly_pred_train
        ypoly_test[i, :] = ypoly_pred_test

    # plot
    plt.figure(2)
    plt.scatter(x_train, y_train, c='black')
    plt.plot(x_test, ypoly_test[0, :], label=f'degree: {d[0]}')
    plt.plot(x_test, ypoly_test[1, :], label=f'degree: {d[1]}')
    plt.plot(x_test, ypoly_test[2, :], label=f'degree: {d[2]}')
    plt.title(f'Polynomial Ridge Regression')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()

    '''for i, d in enumerate([2, 3, 4]):
        ypoly_pred_train, ypoly_pred_test = polynomial_regression(x_train, x_test, y_train, y_test,
                                                      d, k)

        rmsepoly_train = compute_rmse(y_train, ypoly_pred_train)
        rmsepoly_test = compute_rmse(y_test, ypoly_pred_test)
        print(f'RMSE Train with degree: {d}', rmsepoly_train)
        print(f'RMSE Test with degree: {d}', rmsepoly_test)
        # ypoly_pred_train oder ypoly_pred_test: muss train
        plot_polyregression(x_train, y_train, ypoly_pred_train, d)'''




if __name__ == '__main__':
    main()