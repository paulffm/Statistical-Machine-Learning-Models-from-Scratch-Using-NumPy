import numpy as np
from matplotlib import pyplot as plt

def main():
    # LDA as multi class classifier from several 2-class classifier (one vs one)
    data = np.loadtxt("ldaData.txt", dtype=float)
    datac1 = data[:50, :]
    datac2 = data[50:93, :]
    datac3 = data[93:137, :]

    # split data for 2-class classifier
    data12 = data[:93, :]
    data23 = data[50:137, :]
    data13 = np.delete(data, slice(50, 93), axis=0)

    # mean and covariance for each class
    meanc1 = np.mean(datac1, axis=0)[:, np.newaxis].reshape(1, 2)
    covc1 = np.cov(datac1[:, 0], datac1[:, 1]) * (datac1.shape[0] - 1)

    meanc2 = np.mean(datac2, axis=0)[:, np.newaxis].reshape(1, 2)
    covc2 = np.cov(datac2[:, 0], datac2[:, 1]) * (datac2.shape[0] - 1)

    meanc3 = np.mean(datac3, axis=0)[:, np.newaxis].reshape(1, 2)
    covc3 = np.cov(datac3[:, 0], datac3[:, 1]) * (datac3.shape[0] - 1)


    # LDA classification between class 1 and 2
    SW1 = covc1 + covc2
    mean12 = np.mean(data12, axis=0)[:, np.newaxis].reshape(1, 2)
    w12 = np.linalg.inv(SW1).dot((meanc1 - meanc2).T)
    plabel12 = (w12.T @ (data12 - mean12).T).reshape(-1, )

    # if plabel12 > 0: decide for class 1
    plabel12 = (plabel12 > 0) * 1
    # index of points classified as class 1, 2
    pos1_12 = np.where(plabel12 == 1)
    pos0_12 = np.where(plabel12 == 0)
    # number of missclassified points
    n_miss12 = (np.count_nonzero(plabel12[:datac1.shape[0]] == 0)) + \
              (np.count_nonzero(plabel12[datac1.shape[0]:data12.shape[0]] == 1))
    class1_12 = data[pos1_12]
    class2_12 = data[pos0_12]

    # LDA classification between class 1 and 3
    SW13 = covc1 + covc3
    mean13 = np.mean(data13, axis=0)[:, np.newaxis].reshape(1, 2)
    w13 = np.linalg.inv(SW13).dot((meanc1 - meanc3).T)
    plabel13 = (w13.T @ (data13 - mean13).T).reshape(-1, )

    # if plabel13 > 0: decide for class 1
    plabel13 = (plabel13 > 0) * 1

    # index of points classified as class 1, 3
    pos1_13 = np.where(plabel13 == 1)
    pos0_13 = np.where(plabel13 == 0)

    # if I use this classification + classification between class 2 and 3, I get the same result as If I use
    # classification between 1 and 2 + classification between class 2 and 3
    class1_13 = data13[pos1_13]
    class3_13 = data13[pos0_13]

    # number of missclassified points
    n_miss13 = (np.count_nonzero(plabel13[:datac1.shape[0]] == 0)) + \
               (np.count_nonzero(plabel13[datac1.shape[0]:data13.shape[0]] == 1))

    # LDA classification between class 2 and 3
    SW23 = covc2 + covc3
    mean23 = np.mean(data23, axis=0)[:, np.newaxis].reshape(1, 2)
    w23 = np.linalg.inv(SW23).dot((meanc2 - meanc3).T)
    plabel23 = (w23.T @ (data23 - mean23).T).reshape(-1, )

    # if plabel12 > 0: decide for class 2
    plabel23 = (plabel23 > 0) * 1

    # index of points classified as class 2
    pos1_23 = np.where(plabel23 == 1)
    pos0_23 = np.where(plabel23 == 0)

    class2_23 = data23[pos1_23]
    class3_23 = data23[pos0_23]

    # number of missclassified points
    n_miss23 = (np.count_nonzero(plabel23[:datac2.shape[0]] == 0)) + \
               (np.count_nonzero(plabel23[datac2.shape[0]:data23.shape[0]] == 1))


    print('Number of missclassified points:', n_miss23 + n_miss12)

    # plot
    plt.figure(1, figsize=(7, 7))
    plt.subplot(211)
    plt.scatter(datac1[:, 0], datac1[:, 1], c='b', label='class 1')
    plt.scatter(datac2[:, 0], datac2[:, 1], c='g', label='class 2')
    plt.scatter(datac3[:, 0], datac3[:, 1], c='r', label='class 3')
    plt.title('data points true classification')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.subplot(212)
    plt.scatter(class1_12[:, 0], class1_12[:, 1],  c='b', label='pred class 1')
    plt.scatter(class2_12[:, 0], class2_12[:, 1], c='g', label='pred class 2')
    plt.scatter(class2_23[:, 0], class2_23[:, 1], c='g')
    plt.scatter(class3_23[:, 0], class3_23[:, 1], c='r', label='pred class 3')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('data points pred classification')
    plt.legend()
    plt.show()

    '''plt.figure(2, figsize=(7, 5))
    plt.subplot(211)
    plt.scatter(class1_13[:, 0], class1_13[:, 1], c='b', label='pred class 1')
    plt.scatter(class2_23[:, 0], class2_23[:, 1], c='g')
    plt.scatter(class3_23[:, 0], class3_23[:, 1], c='r', label='pred class 3')
    plt.title('data points pred classification')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.subplot(212)
    plt.scatter(class1_12[:, 0], class1_12[:, 1], c='b', label='pred class 1')
    plt.scatter(class2_12[:, 0], class2_12[:, 1], c='g', label='pred class 2')
    plt.scatter(class2_23[:, 0], class2_23[:, 1], c='g')
    plt.scatter(class3_23[:, 0], class3_23[:, 1], c='r', label='pred class 3')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('data points pred classification')
    plt.legend()
    plt.show()'''


if __name__ == '__main__':
    main()