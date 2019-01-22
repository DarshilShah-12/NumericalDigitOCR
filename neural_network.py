import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(sig):
    return sig * (1 - sig)

'''Neural Network Parameters:
    input layer size => 784 
    hidden layer size => 48
    output layer size => 10
    
    Each layer will have x neurons (listed above) with an activation, bias, and weight for each corresponding neuron in
    the previous and next layer. All neurons are connected to all of the neurons in the previous & next layer. The idea
    is to sum the products of each previous neuron's activation with its corresponding weight and pass it through an 
    activation function (sigmoid in our case) to eventually reach the output layer. By backpropagating the error for 
    thousands of samples, the hope is to reduce error and improve accuracy by adjusting the weights and biases.
    
    Neural network takes in a 28x28 image of a handwritten digit, and outputs 10 values between 0-1 (corresponding
    to each possible digit). The goal is to use the MNIST training set of 50000+ images and to be able to reach 
    an accuracy of >90% with the test set using backpropagation.
'''

class NeuralNetwork:
    def __init__(self):
        self.input = np.zeros((784, 1))
        self.weights_1 = 2 * (np.random.rand(48, 784)) - 1
        self.weights_2 = 2 * (np.random.rand(10, 48)) - 1
        self.bias_1 = np.random.rand(48, 1)
        self.bias_2 = np.random.rand(10, 1)
        self.z_1 = np.zeros((48, 1))
        self.z_2 = np.zeros((10, 1))
        self.activation_1 = np.zeros((48, 1))
        self.activation_2 = np.zeros((10, 1))

    def feed_forward(self, input):
        '''
        :param input: input image (28x28 pixels, passed as an ndarray)
        :return: none
        This function passes the input image and feeds forward to the hidden layer and ultimately the output layer
        '''
        self.input = input.flatten().reshape(784, 1)
        self.z_1 = np.dot(self.weights_1, self.input) + self.bias_1
        self.activation_1 = sigmoid(self.z_1)
        self.z_2 = np.dot(self.weights_2, self.activation_1) + self.bias_2
        self.activation_2 = sigmoid(self.z_2)

    def backprop(self, input, label):
        '''
        :param input: input image (28x28 pixels, passed as an ndarray)
        :param label: corresponding label for the image
        :return: none
        backprop calculates an estimation of the error/loss gradient in order to adjust the weights and biases.
        '''
        #TODO: work in progress... does not work
        y = np.zeros((10, 1))
        y[label] = 1
        loss = self.activation_2 - y
        gradient_approx_2 = np.zeros((10, 48))

        for j in range(len(gradient_approx_2)):
            for k in range(len(gradient_approx_2[j])):
                gradient_approx_2[j][k] = 2 * loss[j] * sigmoid_derivative(self.z_2[j]) * self.activation_1[k]

        return gradient_approx_2

    def train(self, inputs, labels):
        #TODO: Also work in progress.
        gradient_approximator_2 = np.zeros((10, 48))
        for i in range(1000):
            self.feed_forward(inputs[i])
            gradient_approximator_2 += self.backprop(inputs[i], labels[i])
        gradient_approximator_2 /= 1000
        self.weights_2 -= gradient_approximator_2

    def open_load(self):
        self.weights_1 = np.loadtxt('weights_1.csv', delimiter=',')
        self.weights_2 = np.loadtxt('weights_2.csv', delimiter=',')
        self.bias_1 = np.loadtxt('bias_1.csv', delimiter=',').reshape(48, 1)
        self.bias_2 = np.loadtxt('bias_2.csv', delimiter=',').reshape(10, 1)

    def save(self):
        np.savetxt('weights_1.csv', self.weights_1, delimiter=',', fmt='%f')
        np.savetxt('weights_2.csv', self.weights_2, delimiter=',', fmt='%f')
        np.savetxt('bias_1.csv', self.bias_1, delimiter=',', fmt='%f')
        np.savetxt('bias_2.csv', self.bias_2, delimiter=',', fmt='%f')

if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    #Show first training image
    # plt.imshow(x_train[0], cmap='Greys')
    # plt.show()

    neural_net = NeuralNetwork()

    neural_net.open_load()

    neural_net.train(x_train, y_train)
    neural_net.feed_forward(x_test[0])
    neural_net.save()



