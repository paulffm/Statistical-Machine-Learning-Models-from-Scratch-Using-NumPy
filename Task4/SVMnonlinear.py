import numpy as np
import matplotlib.pyplot as plt
import cvxopt
from cvxopt import matrix
from cvxopt import solvers

np.random.seed(12345)

def gaussian_kernel(x, y, sigma=5.0):
    return np.exp(-np.linalg.norm(x-y)**2 / (2 * (sigma ** 2)))

def linear_kernel(x1, x2):
    return np.dot(x1, x2.T)

def polynomial_kernel(x, y, p=3):
    return (1 + np.dot(x, y)) ** p



def main():
    datapca = np.loadtxt("iris-pca.txt", dtype=float)
    X = datapca[:, :2]
    label = datapca[:, 2]

    # split data in the 2 classes
    k = np.min(np.argwhere(label == 2))
    # +1, -1 for SVM
    y_pos = label[:k]+1
    y_neg = label[k:]-3
    y = np.stack((y_pos, y_neg), axis=0)
    y = y.reshape(-1, 1)


    m, n = X.shape
    K_gauss = np.zeros((m, m))
    K_lin = np.zeros((m, m))

    for i in range(m):
        for j in range(m):
            K_gauss[i, j] = gaussian_kernel(X[i, :], X[j, :])

    for i in range(m):
        for j in range(m):
            K_lin[i, j] = linear_kernel(X[i, :], X[j, :])

    # penalty term
    C = 0.1

    # matrices
    P = cvxopt.matrix(np.outer(y, y) * K_lin)
    q = cvxopt.matrix(np.ones(m) * -1)
    G = cvxopt.matrix(np.vstack((np.eye(m) * -1, np.eye(m))))
    h = cvxopt.matrix(np.hstack((np.zeros(m), np.ones(m) * C)))
    A = cvxopt.matrix(y.reshape(1, -1))
    b = cvxopt.matrix(np.zeros(1))

    # solving the quadratic problem
    sol = cvxopt.solvers.qp(P, q, G, h, A, b)
    # lagrange multipliers
    alpha = np.array(sol['x'])

    # support vectors >0
    supportv = alpha > 1e-6
    # indices support vector
    X_sv = X[np.squeeze(supportv)]
    idx = np.arange(len(alpha))[:, np.newaxis]
    idx = idx[supportv]

    # b
    b1 = 0
    b2 = 0
    # for all Kernels
    for i in idx:
        b1 += y[i]
        for j in idx:
            b2 += alpha[j] * y[j] * gaussian_kernel(X[i, :], X[j, :])
    b = (b1 - b2) / len(idx)


    x_min = np.min(X[:, 0])
    x_max = np.max(X[:, 0])
    y_min = np.min(X[:, 1])
    y_max = np.max(X[:, 1])

    X1, X2 = np.meshgrid(np.linspace(x_min, x_max), np.linspace(y_min, y_max))
    X_mesh = np.array([[x1, x2] for x1, x2 in zip(np.ravel(X1), np.ravel(X2))])

    # prediction:
    y_predict = np.zeros(len(X))
    for i in range(len(X)):
        s = 0
        for j in idx:
            s += alpha[j] * y[j] * gaussian_kernel(X[i, :], supportv[j])
        y_predict[i] = s

    y_predict = y_predict + b

    y_predict = np.sign(y_predict)[:, np.newaxis]
    # accuracy
    acc = np.sum((y_predict == y) * 1) / len(y)
    print(f'Accuracy: {acc*100}%')

    z = np.zeros(len(X_mesh[:, 0]))
    for i in range(len(X_mesh[:, 0])):
        s = 0
        for j in idx:
            s += alpha[j] * y[j] * gaussian_kernel(X[i, :], supportv[j])
        z[i, :] = s
    Z = z + b


    # indices of missclassified points
    false_idx = np.squeeze((y_predict != y))
    X_false = X[false_idx, :]



    # plot
    plt.figure(figsize=(8, 8))
    plt.contour(X1, X2, Z, colors='k', linewidths=1, origin='lower')
    plt.scatter(X[:k, 0], X[:k, 1], c='blue', label='0')
    plt.scatter(X[k:, 0], X[k:, 1], c='green', label='2')
    plt.scatter(X_false[:, 0], X_false[:, 1], facecolors='none', edgecolors='red', marker='o', label='Missclassified')
    plt.scatter(X_sv[:, 0], X_sv[:, 1], facecolors='none', edgecolors='yellow', marker='o', label='Support Vectors')
    plt.xlim(x_min - 0.1, x_max + 0.1)
    plt.ylim(y_min - 0.1, y_max + 0.1)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()