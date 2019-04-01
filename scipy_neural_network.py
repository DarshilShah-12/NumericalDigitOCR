import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import scipy

def sigmoid(x):
    return 1/(1+np.exp(-1*x))

def first_order_sigmoid_derivative(x):
    return sigmoid(x)*(1-sigmoid(x))

class NeuralNetwork:
    def __init__(self):
        self.input = np.zeros((64, 1))
        self.weights_input_to_hidden_layer = 2*np.random.rand(10, 64) - 1
        self.hidden_layer = np.zeros((48, 1))
        self.z = np.zeros((10, 1))
        self.final_activation_values = np.zeros((10, 1))

    def feed_forward(self, singular_input_matrix):
        self.input = singular_input_matrix.flatten().reshape(64, 1)/15
        self.z = np.dot(self.weights, self.input)
        self.final_activation_values = sigmoid(self.z)

    def back_propagation(self, singular_matrix_label):
        expected_activation_values = np.zeros((10, 1))
        expected_activation_values[singular_matrix_label] = 1
        instance_diff = self.final_activation_values - expected_activation_values
        gradient = np.zeros((10, 64))

        for j in range(len(gradient)):
            for k in range(len(gradient[j])):
                gradient[j][k] = 2*instance_diff[j]*first_order_sigmoid_derivative(self.z[j])*self.input[k]

        return gradient

    def train(self, input_matrices, matrix_labels):
        for l in range(len(input_matrices)):
            self.feed_forward(input_matrices[l])
            self.weights -= self.back_propagation(matrix_labels[l])

    def test_accuracy(self, test_matrices, test_labels):
        correct_tally = 0
        for m in range(len(test_matrices)):
            self.feed_forward(test_matrices[m])
            if np.argmax(self.final_activation_values) == test_labels[m]:
                correct_tally += 1
        print("Runtime Accuracy: " + str(round(correct_tally*100/len(test_matrices), 2)) + "%")

if __name__ == "__main__":
    digits = datasets.load_digits()
    plt.imshow(digits['images'][5], cmap='Greys')
    # plt.show()
    neural_net = NeuralNetwork()

    neural_net.train(digits['images'], digits['target'])

    neural_net.test_accuracy(digits['images'], digits['target'])

# Note: With zero hidden layers and a direct mapping of input to output, test_accuracy hovers from 91% to 94%