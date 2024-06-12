import numpy as np
import matplotlib.pyplot as plt
np.random.seed(12345)

def target_func(x):
    return np.cos(x) + (np.sin(x) ** 2)

def exp_kernel(x1, x2):
    # is assumption o_f, l=1 okay?
    o_f = 1
    l = 1
    return np.array([[(o_f ** 2) * np.exp(- np.linalg.norm(x_i - x_j) ** 2 / (2 * l ** 2)) for x_j in x2] for x_i in x1])

def GP(x1, y1, x2, o_n):
    cov11 = exp_kernel(x1, x1) + o_n ** 2 * np.eye(len(x1))
    cov12 = exp_kernel(x1, x2)
    # is noise for cov22 right?
    cov22 = exp_kernel(x2, x2) + o_n ** 2 * np.eye(len(x2))
    q = np.linalg.inv(cov11) @ cov12
    mu21 = q.T @ y1
    sigma21 = cov22 - q.T @ cov12

    return mu21, sigma21

def plot_GP(x1, x2, y1, y_t, mu2, o2, i):
    if i == 0 or i == 1 or i == 4 or i == 9 or i == 14:
        plt.figure()
        plt.plot(x2, y_t, label='y(x)=cos(x)+sin(x)*sin(x)', color='red')
        plt.plot(x2, mu2, label='mu2|1', color='blue')
        plt.scatter(x1, y1, label='data points', color='black')
        plt.fill_between(x2, mu2 - 2 * o2, mu2 + 2 * o2, color='red', alpha=0.15, label='2o2|1')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'Gaussian Procesess with {i + 1} points')
        plt.legend()
        plt.show()

def main():

    x2 = np.arange(0, 2 * np.pi + 0.005, 0.005)
    y_t = target_func(x2)
    o_n = 0.005
    # I pick random first point, or how to pick first point: start with no target?
    x1 = np.random.choice(x2, 1)
    x1 = x1.tolist()

    for i in range(15):
        # noise verschiebt punkte nicht
        y1 = target_func(x1) + o_n ** 2 * np.random.randn(len(x1))
        mu2, sigma2 = GP(x1, y1, x2, o_n)
        o2 = np.sqrt(np.diag(sigma2))
        plot_GP(x1, x2, y1, y_t, mu2, o2, i)

        k = np.argmax(o2)
        x1.append(x2[k])

if __name__ == '__main__':
    main()