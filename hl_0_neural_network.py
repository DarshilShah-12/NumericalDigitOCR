import keras
import numpy as np
import scipy
from scipy import special

def sigmoid(x):
    return scipy.special.expit(x)

def first_order_sigmoid_derivative(x):
    return sigmoid(x)*(1-sigmoid(x))

class NeuralNetwork:
    def __init__(self):
        self.input = np.zeros((784, 1))
        self.weights = 2 * np.random.rand(10, 784) - 1
        self.z = np.zeros((10, 1))
        self.activation = np.zeros((10, 1))
        self.weights_path = "learnable_parameters/weights/"

    def feed_forward(self, input):
        self.input = input.flatten().reshape(784, 1) / 255
        self.z = np.dot(self.weights, self.input)
        self.activation = sigmoid(self.z - 3)

    def backpropagation(self, label):
        expected = np.zeros((10, 1))
        expected[label] = 1
        diff = self.activation - expected
        d_activation_wr_z = first_order_sigmoid_derivative(self.z)
        d_z_wr_weights = self.input

        d_diff_wr_weights = np.dot(diff * d_activation_wr_z, d_z_wr_weights.T)
        self.weights -= 0.8*d_diff_wr_weights


    def train(self, input_matrices, matrix_labels):
        for l in range(len(input_matrices)):
            self.feed_forward(input_matrices[l])
            self.backpropagation(matrix_labels[l])

    def test_accuracy(self, test_matrices, test_labels):
        correct_tally = 0
        for m in range(len(test_matrices)):
            self.feed_forward(test_matrices[m])
            if np.argmax(self.activation) == test_labels[m]:
                correct_tally += 1
        print("Runtime Accuracy: " + str(round(correct_tally * 100 / len(test_matrices), 2)) + "%")

    def open_load(self):
        self.weights = np.loadtxt(self.weights_path + 'weights.csv', delimiter=',')

    def save(self):
        np.savetxt(self.weights_path + 'weights.csv', self.weights, delimiter=',', fmt='%f')

if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    neural_net = NeuralNetwork()
    neural_net.open_load()
    neural_net.train(x_train, y_train)
    neural_net.test_accuracy(x_test, y_test)
    neural_net.save()