import random
import numpy as np
from matplotlib import pyplot as plt

class Neural_Network:
    """Neural Network of given size which can be trained via stochastic gradient 
    descent."""
    def __init__(self, size):
        '''The parameter 'size' should be a list of integers with the length
        of the list specifying the number of layers and the value of each element 
        specifying the number of neurons in the respective layer. For instance, the 
        list [10, 20, 2] will create a neural network with three layers, with the 
        first layer containing 10 neurons, the second layer 20 neurons and the third
        layer 2 neurons. Thereby, the first layer represents the input layer and the
        last layer represents the output layer.'''
        self.number_of_layers = len(size)
        self.weights = [np.random.randn(rows, cols) for rows, cols in zip(size[1:], size[:-1])]
        self.biases = [np.random.randn(rows, 1) for rows in size[1:]]
    
    def forward_prop(self, x):
        '''The forward propagation of a given input x. Returns the activations of 
        all layers. The last element of 'activations' is the output of the network.'''
        a = x
        activations = [x]
        zets = []
        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            z = np.dot(w, a) + b
            zets.append(z)
            a = ReLU(z)
            activations.append(a)
        # last step uses a different activation function
        z_output = np.dot(self.weights[-1], activations[-1]) + self.biases[-1]
        zets.append(z_output)
        a_output = sigmoid(z_output)
        activations.append(a_output)
        return zets, activations

    def train(self, training_data, iterations, mini_batch_size, learning_rate):
        '''Trains the network via stochastic gradient descent. The parameter 
        'training_data' should be a list of tuples of the form (x, y) with x 
        representing the inputs and y the corresponding desired outputs. '''
        for i in range(iterations):
            costs = 0
            number_correct_predictions = 0
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, len(training_data), mini_batch_size)]
            for mini_batch in mini_batches:
                costs_mini_batch, number_correct_predictions_mini_batch = self.gradient_descent(mini_batch, learning_rate)
                costs += costs_mini_batch
                number_correct_predictions += number_correct_predictions_mini_batch
            print("Iteration: ", i+1)
            print("Correct Predictions: ", number_correct_predictions)
            print("Accuracy: " + str(number_correct_predictions/60000 * 100) + "%")
            print("Costs: ", 1/60000 * costs)
            print()

    def gradient_descent(self, mini_batch, learning_rate):
        '''Adjusts the weights and biases of the network with the summed up
        deltas of weights and biases for one mini batch. The deltas represent
        the gradient of the cost function and are calculated via backpropagation.'''
        delta_weights, delta_biases, costs, number_correct_predictions = self.backprop(mini_batch)
        self.weights = [w-(learning_rate/len(mini_batch))*dw for w, dw in zip(self.weights, delta_weights)]
        self.biases = [b-(learning_rate/len(mini_batch))*db for b, db in zip(self.biases, delta_biases)]
        return costs, number_correct_predictions

    def backprop(self, mini_batch):
        '''Calculates and sums up the deltas of weigths and biases for one mini batch.
        The deltas represent the gradient of the cost function.'''
        delta_weights = [np.zeros(w.shape) for w in self.weights]
        delta_biases = [np.zeros(b.shape) for b in self.biases]
        costs = 0
        number_correct_predictions = 0
        for x, y in mini_batch:
            zets, activations = self.forward_prop(x)
            # calculate the error and the value of the cost function
            error = activations[-1] - y
            cost_vec = np.square(error)
            costs += np.sum(cost_vec)
            # backprop and calculate the deltas of weights and biases
            dz = 2 * error * sigmoid_prime(zets[-1])
            delta_weights[-1] += np.dot(dz, activations[-2].T)
            delta_biases[-1] += dz
            for i in range(2, self.number_of_layers):
                dz = np.dot(self.weights[-i+1].T, dz) * ReLU_prime(zets[-i])
                delta_weights[-i] += np.dot(dz, activations[-i-1].T)
                delta_biases[-i] += dz
            # raise the number of correct predictions if the output of 
            # the network matches the desired output
            if (np.argmax(activations[-1]) == np.argmax(y)):
                number_correct_predictions += 1
        return delta_weights, delta_biases, costs, number_correct_predictions

    def test(self, test_data):
        '''Test the neural network with the given test data.'''
        for _ in range(10):
            i = random.randrange(len(test_data))
            x, y = test_data[i]
            print("Label: ", y)
            plt.gray()
            plt.imshow(x)
            plt.show()
            x = x.flatten().reshape(784, 1)
            activations = self.forward_prop(x)[-1]
            print("Output :")
            print(np.round(activations[-1], 3))
            print("Prediction: ", np.argmax(activations[-1]))
            print()
            print()


# Different activation functions and their derivatives
def ReLU(Z):
    relu = np.maximum(0, Z)
    max_value = np.max(relu)
    if max_value != 0:
        return relu/max_value
    return relu

def ReLU_prime(Z):
    return Z > 0

def softmax(Z):
    e_Z = np.exp(Z - Z.max())
    return e_Z / e_Z.sum()

def sigmoid(Z):
    return 1.0/(1.0+np.exp(-Z))

def sigmoid_prime(Z):
    return sigmoid(Z)*(1-sigmoid(Z))