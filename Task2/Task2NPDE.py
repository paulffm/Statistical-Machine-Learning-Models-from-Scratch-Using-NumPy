import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import KernelDensity

# non parametric density estimation
# Histogram, KDE, KNN, loglikelihood without given distribution, comparison
# expectation and maximization
# all from scratch

def KDE(x, data, h):

    y = np.zeros((len(x)))
    for i in range(len(x)):
        hold = 0
        for xi in range(len(data)):
            hold += np.exp(- ((x[i]-data[xi]) ** 2) / (2 * h * h))
        y[i] = hold * 1 / (np.sqrt(2 * np.pi * h * h) * len(data))
    return y


def plothist(data, width):
    # plot histograms with 3 different widths
    n1 = np.ceil((np.max(data) - np.min(data)) / width[0])
    n1 = int(n1)


    n2 = np.ceil((np.max(data) - np.min(data)) / width[1])
    n2 = int(n2)


    n3 = np.ceil((np.max(data) - np.min(data)) / width[2])
    n3 = int(n3)

    plt.subplot(131)
    plt.hist(data, bins=n1)
    plt.title('width = 0.02')
    plt.subplot(132)
    plt.hist(data, bins=n2)
    plt.title('width = 0.5')
    plt.subplot(133)
    plt.hist(data, bins=n3)
    plt.title('width = 2')
    plt.show()
    plt.legend()

def euclidian(v1, v2):
    dist = np.sqrt(np.sum((v1 - v2) ** 2))
    return dist

def knearestneighbour(x_train, x_test, k):
    n = len(x_train)
    density = np.zeros(len(x_test))

    for j in range(len(x_test)):
        # Array to store distances
        # point_dist = np.zeros(len(x_train))
        point_dist = []

        for i in range(len(x_train)):
            distances = np.abs((x_train[i] - x_test[j]))
            # point_dist[i] = distances
            point_dist.append(distances)

        point_distarr = np.asarray(point_dist)
        point_distarr = np.sort(point_distarr)
        # neighbour_k = neighbours[k-1]
        # neighbours = point_dist.argsort()
        # neighbour_k = neighbours[k]
        # density[j] = k / (len(x_train) * 2 * point_dist[neighbour_k])
        density[j] = (k / n) * 1 / (2 * point_distarr[k])
    return density

def expectation_max(data, max_iter=1000):
    #data = pd.DataFrame(data)
    mu0 = data.mean()
    c0 = data.cov()

    for j in range(max_iter):
        w = []
        # perform the E part of algorithm
        for i in data:
            wk = (5 + len(data))/(5 + np.dot(np.dot(np.transpose(i - mu0), np.linalg.inv(c0)), (i - mu0)))
            w.append(wk)
            w = np.array(w)

        # perform the M part of the algorithm
        mu = (np.dot(w, data))/(np.sum(w))

        c = 0
        for i in range(len(data)):
            c += w[i] * np.dot((data[i] - mu0), (np.transpose(data[i] - mu0)))
        cov = c/len(data)

        mu0 = mu
        c0 = cov

    return mu0, c0



def main():

    testdata = np.loadtxt("nonParamTest.txt", dtype=float)
    testdsort = np.sort(testdata)
    traindata = np.loadtxt("nonParamTrain.txt", dtype=float)
    traindsort = np.sort(traindata)

    # plot histograms
    width = [0.02, 0.5, 2]
    # plothist(testdata, width)


    # knn:
    ka = [2, 10, 35]
    densityknn_test = np.zeros((3, len(testdata)))
    densityknn_train = np.zeros((3, len(traindata)))
    for i, ki in enumerate(ka):
        densityknn_test[i, :] = knearestneighbour(traindata, testdsort, ki)
        densityknn_train[i, :] = knearestneighbour(traindata, traindsort, ki)

    # plt.plot(testdsort, densityknn_test)
    # knn train plot
    plt.figure(2)
    plt.subplot(131)
    plt.plot(traindsort, densityknn_train[0, :])
    plt.xlim(-4, 8)
    plt.xlabel('x')
    plt.ylabel('prob')
    plt.title(f'$k=2$')
    plt.subplot(132)
    plt.plot(traindsort, densityknn_train[1, :])
    plt.xlim(-4, 8)
    plt.xlabel('x')
    plt.ylabel('prob')
    plt.title(f'$k=10$')
    plt.subplot(133)
    plt.plot(traindsort, densityknn_train[2, :])
    plt.xlim(-4, 8)
    plt.xlabel('x')
    plt.ylabel('prob')
    plt.title(f'$k=35$')
    plt.show()

    # knn test plot
    plt.figure(3)
    plt.subplot(131)
    plt.plot(testdsort, densityknn_test[0, :])
    plt.xlim(-4, 8)
    plt.xlabel('x')
    plt.ylabel('prob')
    plt.title(f'$k=2$')
    plt.subplot(132)
    plt.plot(testdsort, densityknn_test[1, :])
    plt.xlim(-4, 8)
    plt.xlabel('x')
    plt.ylabel('prob')
    plt.title(f'$k=10$')
    plt.subplot(133)
    plt.plot(testdsort, densityknn_test[2, :])
    plt.xlim(-4, 8)
    plt.xlabel('x')
    plt.ylabel('prob')
    plt.title(f'$k=35$')
    plt.show()


    # gaussian kernel
    h = [0.03, 0.2, 0.8]
    x_d = np.linspace(-4, 8, 500)

    denskde_test = np.zeros((3, len(x_d)))
    denskde_train = np.zeros((3, len(x_d)))

    for i, hi in enumerate(h):
        denskde_train[i, :] = KDE(x_d, traindata, hi)
        denskde_test[i, :] = KDE(x_d, testdata, hi)

    # kde plot traindata
    plt.figure(4)
    plt.plot(x_d, denskde_train[0, :], label='h=0.03')
    plt.plot(x_d, denskde_train[1, :], label='h=0.2')
    plt.plot(x_d, denskde_train[2, :], label='h=0.8')
    plt.xlim(-4, 8)
    plt.xlabel('x')
    plt.ylabel('p')
    plt.title('KDE Density estimation traindata')
    plt.show()

    # kde plot testdata
    plt.figure(5)
    plt.plot(x_d, denskde_test[0, :], label='h=0.03')
    plt.plot(x_d, denskde_test[1, :], label='h=0.2')
    plt.plot(x_d, denskde_test[2, :], label='h=0.8')
    plt.xlim(-4, 8)
    plt.xlabel('x')
    plt.ylabel('p')
    plt.title('KDE Density estimation testdata')
    plt.show()



    # task 3d: compute loglikelihood and compare
    # to compare if
    lllknn_train = np.zeros(2)
    lllknn_test = np.zeros(2)
    lllkde_train = np.zeros(2)
    lllkde_test = np.zeros(2)

    # o=h= 0.03 -> divide by zero error
    for i in range(2):
        lllknn_train[i] = np.sum(np.log(densityknn_train[i+1, :]))
        lllknn_test[i] = np.sum(np.log(densityknn_test)[i+1, :])
        lllkde_train[i] = np.sum(np.log(denskde_train[i+1, :]))
        lllkde_test[i] = np.sum(np.log(denskde_test[i+1, :]))

    print(lllknn_train)
    print(lllknn_test)
    print(lllkde_train)
    print(lllkde_test)

if __name__ == '__main__':
    main()