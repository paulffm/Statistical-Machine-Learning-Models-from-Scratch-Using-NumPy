import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d

# parametric density estimation
# bivariate gaussian, loglikelihood, posterior, decision boundary from scratch
def selfmean(data):
    sum = 0
    for i in data:
        sum += i
    return sum / len(data)
def multivariate_normal(x, d, mean, covariance):
    """pdf of the multivariate normal distribution."""
    x_m = x - mean
    return (1. / (np.sqrt((2 * np.pi) ** d * np.linalg.det(covariance))) *
            np.exp(-((x_m.T @ np.linalg.inv(covariance)).dot(x_m)) / 2))
def selfcovariance(data):
    datax = data[:, 0]
    datay = data[:, 1]
    meanx = selfmean(datax)
    meany = selfmean(datay)
    varxx = 0
    varyy = 0
    varxy = 0
    for i in range(len(datax)):
        varxx += (datax[i] - meanx) ** 2
        varyy += (datay[i] - meany) ** 2
        varxy += (datax[i] - meanx) * (datay[i] - meany)
    cov = np.array([[varxx, varxy], [varxy, varyy]])
    return cov / len(datax)

def bivariate_normal(x, y, mean, cov):
    """pdf of the bivariate normal distribution."""
    oxx = np.sqrt(cov[0, 0])
    oyy = np.sqrt(cov[1, 1])
    varxy = cov[0, 1]
    x_m = x - mean[0]
    y_m = y - mean[1]
    corr = varxy / (oxx * oyy)
    first = 1. / (np.sqrt(1-corr ** 2) * 2 * np.pi * oxx * oyy)
    print(first.shape)
    second = np.exp(- 1 / (2 * (1 - corr ** 2)) * ((x_m / oxx) ** 2 - 2 * corr * (x_m / oxx) * (y_m / oyy) + (y_m / oyy) ** 2))
    return first * second

def bivariate_normal2(x, y, mean, cov):
    x_m = x - mean[0]
    y_m = y - mean[1]

    d_m = np.concatenate((x_m, y_m), axis=1)
    print(d_m.shape)
    first = (1. / (np.sqrt((2 * np.pi) ** 2 * np.linalg.det(cov))))
    print(first.shape)
    second = np.exp(-((d_m @ np.linalg.inv(cov) @ d_m.T) / 2))
    return first * second

def decision_boundary(x_vec1, mu1, mu2, cov1, cov2, prior1, prior2):

    mu1 = mu1.reshape(2, 1)
    x_vec1 = np.asarray(x_vec1)
    x_vec1 = x_vec1.reshape(2, -1)
    mu2 = mu2.reshape(2, 1)

    W_1 = (-1 / 2) * np.linalg.inv(cov1)
    w_1 = np.linalg.inv(cov1) @ mu1
    first1 = (((-1 / 2) * mu1.T).dot(np.linalg.inv(cov1))).dot(mu1)
    second1 = (-1 / 2) * np.log(np.linalg.det(cov1))
    omega_1 = first1 + second1 + np.log(prior1)
    g1 = x_vec1.T @ W_1 @ x_vec1 + w_1.T  @ x_vec1 + omega_1

    W_2 = (-1 / 2) * np.linalg.inv(cov2)
    w_2 = np.linalg.inv(cov2) @ mu2
    first2 = (((-1 / 2) * mu2.T).dot(np.linalg.inv(cov2))).dot(mu2)
    second2 = (-1 / 2) * np.log(np.linalg.det(cov2))
    omega_2 = first2 + second2 + np.log(prior2)
    g2 = x_vec1.T @ W_2 @ x_vec1 + w_2.T @ x_vec1 + omega_2

    return g1 - g2

def main():
    data1 = np.loadtxt("densEst1.txt", dtype=float)
    data2 = np.loadtxt("densEst2.txt", dtype=float)
    data = np.concatenate((data1, data2))

    # the mean is unbiased
    mean1 = np.asarray(selfmean(data1))
    mean2 = selfmean(data2)

    # biased covariance
    cov1b = selfcovariance(data1)
    cov2b = selfcovariance(data2)

    # unbiased covariance
    cov1ub = cov1b * len(data1) / (len(data1) - 1)
    cov2ub = cov2b * len(data2) / (len(data2) - 1)

    # class density
    n = 1000
    X1 = np.linspace(-10, 6, n).reshape(-1, 1)
    Y1 = np.linspace(-6, 5, n).reshape(-1, 1)
    X1, Y1 = np.meshgrid(X1, Y1)
    Z1 = bivariate_normal(X1, Y1, mean1, cov1ub)

    X2 = np.linspace(-5, 11, n).reshape(-1, 1)
    Y2 = np.linspace(-2, 9, n).reshape(-1, 1)
    X2, Y2 = np.meshgrid(X2, Y2)
    Z2 = bivariate_normal(X2, Y2, mean2, cov2ub)

    plt.figure(1)
    plt.scatter(data1[:, 0], data1[:, 1], label='dataset1')
    plt.contour(X1, Y1, Z1, 20, cmap='RdGy')
    plt.contour(X2, Y2, Z2, cmap='RdGy')
    plt.scatter(data2[:, 0], data2[:, 1], label='dataset2')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()

    # posterior:
    # prior
    priorc1 = len(data1) / (len(data1) + len(data2))
    priorc2 = len(data2) / (len(data1) + len(data2))

    # posterior: p(C=c1, x) = pr(C=C1) * p(x, C=c1) / p(x) = p(C=c1) * p(x0, C=c1) * p(x1, C=c1) /p(x)

    X3 = np.linspace(-10, 11, n)
    Y3 = np.linspace(-6, 9, n)

    # X3db, Y3db needed to plot the decision boundary
    X3db = X3
    Y3db = Y3

    X3, Y3 = np.meshgrid(X3, Y3)
    Z1 = bivariate_normal(X3, Y3, mean1, cov1ub)
    Z2 = bivariate_normal(X3, Y3, mean2, cov2ub)
    # calucalte the posterior
    post1 = Z1 * priorc1 / (Z1 * priorc1 + Z2 * priorc2)
    post2 = Z2 * priorc2 / (Z1 * priorc1 + Z2 * priorc2)

    ax = plt.axes(projection='3d')
    # to plot decision boundary, we need extra calculation
    Z1db = bivariate_normal(X3db, Y3db, mean1, cov1ub)
    Z2db = bivariate_normal(X3db, Y3db, mean2, cov2ub)
    post1db = Z1db * priorc1 / (Z1db * priorc1 + Z2db * priorc2)
    post2db = Z2db * priorc2 / (Z1db * priorc1 + Z2db * priorc2)
    dboundarydb = np.log(post1db / post2db)

    # plot
    ax.plot_surface(X3, Y3, post1, linewidth=0, antialiased=True, zorder=0.5)
    ax.plot_surface(X3, Y3, post2, linewidth=0, alpha=0.5, antialiased=True, zorder=0.5)
    ax.plot(X3db, Y3db, dboundarydb, label='decision boundary')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('p')
    ax.set_zlim(0, 1.0)
    ax.legend()
    plt.show()


if __name__ == '__main__':
        main()