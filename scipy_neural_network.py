import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import scipy

def sigmoid(x):
    # return 1/(1+np.exp(-1*x))
    return scipy.special.expit(x)

def first_order_sigmoid_derivative(x):
    return sigmoid(x)*(1-sigmoid(x))
    # return np.exp(x/1000)/(1000*(np.exp(x/1000) + 1)**2)

class NeuralNetwork:
    def __init__(self):
        self.input = np.zeros((64, 1))
        self.weights = np.random.rand(10, 64)
        # self.weights = np.zeros((10, 64))
        self.z = np.zeros((10, 1))
        self.final_activation_values = np.zeros((10, 1))

    def feed_forward(self, singular_input_matrix):
        self.input = singular_input_matrix.flatten().reshape(64, 1)/15
        self.z = np.dot(self.weights, self.input) - 10
        self.final_activation_values = sigmoid(self.z)

    def back_propagation(self, singular_matrix_label):
        expected_activation_values = np.zeros((10, 1))
        expected_activation_values[singular_matrix_label] = 1
        instance_diff = self.final_activation_values - expected_activation_values
        # Divide by 0.999999 and see what the domain of the sigmoid function is here
        gradient = np.zeros((10, 64))

        for j in range(len(gradient)):
            for k in range(len(gradient[j])):
                gradient[j][k] = 2*instance_diff[j]*first_order_sigmoid_derivative(self.z[j])*self.input[k]

        return gradient

    def train(self, input_matrices, matrix_labels):
        # gradient_approximator = np.zeros((10, 64))
        for l in range(1700):
            self.feed_forward(input_matrices[l])
            self.weights -= self.back_propagation(matrix_labels[l])

        # gradient_approximator = gradient_approximator/100
        # self.weights += gradient_approximator
        self.feed_forward(input_matrices[1796])
        self.final_activation_values = self.final_activation_values.round(2)
        print(self.final_activation_values)

if __name__ == "__main__":
    digits = datasets.load_digits()
    plt.imshow(digits['images'][6], cmap='Greys')
    # plt.show()
    neural_net = NeuralNetwork()

    print(digits['target'][999])

    neural_net.train(digits['images'], digits['target'])