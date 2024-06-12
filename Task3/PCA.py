import numpy as np
from matplotlib import pyplot as plt


def compute_rmse(y, y_pred):
    return (np.mean((y - y_pred) ** 2, axis=0)) ** 0.5

def main():
    data_iris = np.loadtxt("iris.txt", delimiter=',', skiprows=1)
    X = data_iris[:, 0:4]
    target = data_iris[:, 4]

    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X_norm = (X - mean) / std
    print(X.shape)

    # covariance
    cov_mat = np.cov(X_norm, rowvar=False) * (X_norm.shape[0] - 1 / X_norm.shape[0])

    # eigenvalues, eigenvectors
    eigen_value, eigen_vector = np.linalg.eig(cov_mat)
    # sorted
    idxs = np.argsort(abs(eigen_value))[::-1]
    eigen_value = eigen_value[idxs]
    eigen_vector = eigen_vector[:, idxs]

    # explained variance
    expl_var = eigen_value / np.sum(eigen_value)
    cum_sum_var = np.cumsum(expl_var)

    '''# plot
    plt.figure(1)
    plt.step(range(0+1, len(cum_sum_var)+1), cum_sum_var, where='mid')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal component index')
    plt.tight_layout()
    plt.show()'''

    # need 2 components to explain at least 95% of variance
    k = 2

    eigen_vector_red = eigen_vector[:, 0:k]
    x_red = np.dot(X_norm, eigen_vector_red)

    setosa_idx = np.where(target == 0)
    versi_idx = np.where(target == 1)
    virg_idx = np.where(target == 2)

    '''plt.figure(2)
    plt.scatter(x_red[setosa_idx][:, 0], x_red[setosa_idx][:, 1], label='Setosa')
    plt.scatter(x_red[versi_idx][:, 0], x_red[versi_idx][:, 1], label='Versicolour')
    plt.scatter(x_red[virg_idx][:, 0], x_red[virg_idx][:, 1], label='Virgina')
    plt.xlabel('sepal length')
    plt.ylabel('petal width')
    plt.legend()
    plt.title('Reduced Dimension Plot')
    plt.show()'''


    # reconstruction
    num_pc = [1, 2, 3, 4]
    nrmse = np.zeros((len(num_pc), len(num_pc)))

    '''for i, ki in enumerate(num_pc):
        eigen_vector_red = eigen_vector[:, 0:ki]
        x_red = np.dot(data_norm, eigen_vector_red)
        x_rec = (x_red @ eigen_vector_red.T + mean) * std
        x_rec_max = np.max(x_rec, axis=0)
        x_rec_min = np.min(x_rec, axis=0)
        nrmse[i, :] = (compute_rmse(X, x_rec)) / (x_rec_max - x_rec_min)'''

    for i, ki in enumerate(num_pc):
        eigen_vector_red = eigen_vector[:, 0:ki]
        x_red = np.dot(X_norm, eigen_vector_red)
        x_rec = (x_red @ eigen_vector_red.T)
        x_rec_max = np.max(x_rec, axis=0)
        x_rec_min = np.min(x_rec, axis=0)
        nrmse[i, :] = (compute_rmse(X_norm, x_rec)) / (x_rec_max - x_rec_min)


if __name__ == '__main__':
    main()
