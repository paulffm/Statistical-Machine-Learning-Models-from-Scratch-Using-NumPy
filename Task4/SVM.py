import numpy as np
import matplotlib.pyplot as plt
import cvxopt


def linear_kernel(x1, x2):
    return np.dot(x1, x2.T)

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
    K_lin = np.zeros((m, m))

    for i in range(m):
        for j in range(m):
            K_lin[i, j] = linear_kernel(X[i, :], X[j, :])

    # penalty term
    C = 100

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
    supportv = alpha > 1e-4
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
            b2 += alpha[j] * y[j] * linear_kernel(X[i, :], X[j, :])
    b = (b1 - b2) / len(idx)

    # W linear:
    w = np.zeros(2)
    for i in idx:
        w += alpha[i] * y[i] * X[i, :]

    print('W:', w)
    print('b:', b)

    # predictions
    y_predict = np.sign(X @ w + b)[:, np.newaxis]

    # indices of missclassified points
    false_idx = np.squeeze((y_predict != y))
    X_false = X[false_idx, :]

    # accuracy
    acc = np.sum((y_predict == y) * 1) / len(y)
    print(f'Accuracy: {acc*100}%')

    x_min = np.min(X[:, 0])
    x_max = np.max(X[:, 0])
    y_min = np.min(X[:, 1])
    y_max = np.max(X[:, 1])
    # linear
    dp = np.linspace(x_min, x_max)
    a = -w[0] / w[1]
    yy = a * dp - b / w[1]
    margin = 1 / np.sqrt(np.sum(w ** 2))
    yy_neg = yy - np.sqrt(1 + a ** 2) * margin
    yy_pos = yy + np.sqrt(1 + a ** 2) * margin


    plt.figure()
    plt.plot(dp, yy, c='black')
    plt.plot(dp, yy_neg, "m--")
    plt.plot(dp, yy_pos, "m--")
    plt.scatter(X[:k, 0], X[:k, 1], c='blue', label='class 0')
    plt.scatter(X[k:, 0], X[k:, 1], c='green', label='class 2')
    plt.scatter(X_false[:, 0], X_false[:, 1], facecolors='none', edgecolors='red', s=60 , marker='o', label='Missclassified')
    plt.scatter(X_sv[:, 0], X_sv[:, 1], facecolors='none', edgecolors='yellow', marker='o', label='Support Vectors')
    plt.xlim(x_min - 0.1, x_max + 0.1)
    plt.ylim(y_min - 0.1, y_max + 0.1)
    plt.title(f'Linear SVM with Accuracy: {acc*100}%')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()