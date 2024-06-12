import numpy as np
import matplotlib.pyplot as plt
np.random.seed(12345)

def initialize(input_dim, hidden1_dim, hidden2_dim, output_dim, batch_size):
    W1 = np.random.randn(hidden1_dim, input_dim) * 0.01
    b1 = np.zeros((hidden1_dim, ))
    W2 = np.random.randn(hidden2_dim, hidden1_dim) * 0.01
    b2 = np.zeros((hidden2_dim, ))
    W3 = np.random.randn(output_dim, hidden2_dim) * 0.01
    b3 = np.zeros((output_dim, ))
    parameters = [W1, b1, W2, b2, W3, b3]
    return parameters

def sigmoid(x):
    return 1 / (1+np.exp(-x))

def dsigmoid(x):
    # input has to be sigmoid
    # in backpropagation you insert hidden_i
    return x * (1 - x)

def leaky_Relu(x):
    alpha = 0.1
    idx = x <= 0
    x[idx] = alpha * x[idx]
    return x

def dleaky_Relu(x):
    alpha = 0.1
    idx0 = (x <= 0)
    idx1 = (x > 0)
    x[idx0] = alpha
    x[idx1] = 1
    return x

def loss(prediction, target):
    y = np.sum(0.5 * (prediction - target) ** 2)
    return (1 / target.shape[1]) * np.mean(y)

def dloss(prediction, target):
    return (1 / target.shape[1]) * (prediction - target)

# helper functions
def convert_to_1d_vector(parameters):
    W1, b1, W2, b2, W3, b3 = parameters
    params = np.concatenate([W1.ravel(), b1.ravel(),
                             W2.ravel(), b2.ravel(),
                             W3.ravel(), b3.ravel()], axis=0)

    return params

def convert_to_list(params, input_dim, hidden1_dim, hidden2_dim, output_dim):
    base_idx = 0

    W1 = np.reshape(params[base_idx: base_idx + input_dim * hidden1_dim],
                    (hidden1_dim, input_dim))
    base_idx += input_dim * hidden1_dim

    b1 = params[base_idx: base_idx + hidden1_dim]
    base_idx += hidden1_dim

    W2 = np.reshape(params[base_idx: base_idx + hidden1_dim * hidden2_dim],
                    (hidden2_dim, hidden1_dim))
    base_idx += hidden1_dim * hidden2_dim

    b2 = params[base_idx: base_idx + hidden2_dim]
    base_idx += hidden2_dim

    W3 = np.reshape(params[base_idx: base_idx + hidden2_dim * output_dim],
                    (output_dim, hidden2_dim))
    base_idx += hidden2_dim * output_dim

    b3 = params[base_idx: base_idx + output_dim]

    parameters = [W1, b1, W2, b2, W3, b3]

    return parameters

def visualize(X,y,name):
    N = X.shape[0]
    end = np.round(np.random.rand() * N).astype('int')
    start = end - 5
    plt.figure(figsize=(20, 4))
    for index, (image, label) in enumerate(zip(X[start:end], y[start:end])):
        plt.subplot(1, 5, index + 1)
        plt.imshow(np.reshape(image, (28,28)), cmap=plt.cm.gray)
        plt.title(name + ': ' + str(label), fontsize = 20)
    plt.show()

def vectorized_results(y):
    v = np.zeros(10, dtype=int)
    v[y] = 1.0
    return v

def devectorize_results(y):
    return np.argmax(y)


def update_mini_batch(net, mini_batch, learning_rate):

    N = mini_batch.shape[1]
    y = mini_batch[:, N - net.output_dim:N].T
    x = mini_batch[:, 0:N - net.output_dim].T

    activations = net.forward(x)
    loss_value = loss(activations, y)
    dParameters = net.backward(targets=y)
    net.parameters = net.parameters - learning_rate * dParameters

    return net.parameters, loss_value

# Train neural network with stochastic gradient descent.
def train(net, train_data, label_data, learning_rate):
    # data = np.concatenate([train_data, label_data], axis=1)

    activations = net.forward(train_data)
    loss_value = loss(activations, label_data)
    dParameters = net.backward(targets=label_data)
    net.parameters = net.parameters - learning_rate * dParameters

    return net.parameters, loss_value


# Train neural network with stochastic gradient descent.
def train_batch(net, train_data, label_data, epochs, batch_size, learning_rate):
    N = len(train_data)
    data = np.concatenate([train_data, label_data], axis=1)

    losses = list()
    for i in range(epochs):
        # stochastic mini batch
        np.random.shuffle(data)
        # divide data set into batch_size/N parts
        mini_batches = [data[j:j + batch_size] for j in range(0, N, batch_size)]
        for mini_batch in mini_batches:
            _, loss_value = update_mini_batch(net, mini_batch, learning_rate)
            losses.append(loss_value)

        # print ('Epoch {0} complete'.format(i))

    return net.parameters, losses


class NeuralNet(object):
    def __init__(self, batch_size, input_dim, hidden1_dim, hidden2_dim, output_dim):
        self.batch_size = batch_size

        # size of layers
        self.input_dim = input_dim
        self.hidden_dim_1 = hidden1_dim
        self.hidden_dim_2 = hidden2_dim
        self.output_dim = output_dim

        self.placeholder_in = np.ones((self.input_dim, self.batch_size))
        self.placeholder_latent1 = np.ones((self.hidden_dim_1, self.batch_size))
        self.placeholder_latent2 = np.ones((self.hidden_dim_2, self.batch_size))
        self.placeholder_out = np.ones((self.output_dim, self.batch_size))

        self.parameters = initialize(input_dim, hidden1_dim, hidden2_dim, output_dim, batch_size)

        self.W1, self.b1, self.W2, self.b2, self.W3, self.b3 = self.parameters
        self.parameters = convert_to_1d_vector(self.parameters)

    def forward(self, x):

        #   Computing forward pass
        # input activations
        self.placeholder_in = np.zeros((self.input_dim, x.shape[1]))

        # hidden layer activations
        self.placeholder_latent1 = np.zeros((self.hidden_dim_1, x.shape[1]))
        self.placeholder_latent2 = np.zeros((self.hidden_dim_2, x.shape[1]))

        # output activation
        self.placeholder_out = np.zeros((self.output_dim, x.shape[1]))

        self.placeholder_in = x
        self.placeholder_latent1 = leaky_Relu(np.dot(self.W1, self.placeholder_in) + self.b1[:, np.newaxis])
        self.placeholder_latent2 = leaky_Relu(np.dot(self.W2, self.placeholder_latent1) + self.b2[:, np.newaxis])
        self.placeholder_out = self.W3 @ self.placeholder_latent2 + self.b3[:, np.newaxis]

        return self.placeholder_out

    def backward(self, targets):
        [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3] = convert_to_list(self.parameters, self.input_dim,
                                                                                 self.hidden_dim_1, self.hidden_dim_2,
                                                                                 self.output_dim)
        delta = dloss(self.placeholder_out, targets)

        dw3 = np.dot(delta, self.placeholder_latent2.T)
        db3 = np.sum(delta, axis=1)

        delta = np.dot(self.W3.T, delta) * dleaky_Relu(self.placeholder_latent2)

        dw2 = np.dot(delta, self.placeholder_latent1.T)
        db2 = np.sum(delta, axis=1)

        delta = np.dot(self.W2.T, delta) * dleaky_Relu(self.placeholder_latent1)

        dw1 = np.dot(delta, self.placeholder_in.T)
        db1 = np.sum(delta, axis=1)

        dParameters = convert_to_1d_vector([dw1, db1, dw2, db2, dw3, db3])

        return dParameters


    def predict(self, x, parameters):
        """Predict a test data set on the trained parameters."""
        self.W1, self.b1, self.W2, self.b2, self.W3, self.b3 = convert_to_list(parameters, self.input_dim,
                                                                                 self.hidden_dim_1,
                                                                                 self.hidden_dim_2, self.output_dim)
        return self.forward(x)


def main():
    batch_size = 1
    epochs = 10
    learning_rate = 0.01
    X_train = np.loadtxt("mnist_small_train_in.txt",  delimiter=',', dtype=float)
    X_test = np.loadtxt("mnist_small_test_in.txt", delimiter=',', dtype=float)
    y_train = np.loadtxt("mnist_small_train_out.txt", delimiter=',', dtype=int)
    y_test = np.loadtxt("mnist_small_test_out.txt", delimiter=',', dtype=int)

    # bring labels into one-hot-coded vector form
    expected = np.array([vectorized_results(y) for y in y_train])

    hidden1_dim = 20
    hidden2_dim = 20
    output_dim = 10

    net = NeuralNet(batch_size=batch_size, input_dim=784, hidden1_dim=hidden1_dim, hidden2_dim=hidden2_dim,
                    output_dim=output_dim)

    # Training
    parameters, losses = train_batch(net, X_train, expected, epochs, batch_size, learning_rate=learning_rate)
    #parameters, losses = train(net, X_train, expected, learning_rate=learning_rate)

    print('The loss decreasing during training:')
    plt.figure(figsize=(12, 8))
    plt.plot(losses)
    plt.ylabel('Missclassification Rate')
    plt.xlabel('epochs')
    plt.show()

    # Prediction
    y_pred = net.predict(X_test.T, parameters).T

    y_exp = y_test
    y_pred = [devectorize_results(y) for y in y_pred]
    acc = np.count_nonzero((y_exp == y_pred) * 1) / len(y_pred)

    print(f'Accuracy: {acc * 100}%')



if __name__ == '__main__':
    main()