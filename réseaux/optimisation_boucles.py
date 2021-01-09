import numpy as np
import time


# main class
class MLNeuralNetwork:

    def __init__(self, structure):
        """
        initialise the network, with random weights and bias between [-1, 1]
        :param structure: list = [nbre_input, nbre_hidden1, nbre_hidden2, ..., nbre_output]
                          OR file to load weights and bias
        """

        if type(structure) == list:
            self.layers = len(structure)
            self.structure = np.array(structure)
            self.weights_matrices = []
            self.bias_vectors = []
            for i in range(self.layers - 1):
                weight_matrix = (np.random.rand(structure[i+1], structure[i])) * 2 - 1
                bias_vector = (np.random.random_sample(structure[i + 1])) * 2 - 1
                self.weights_matrices.append(weight_matrix)
                self.bias_vectors.append(bias_vector)
            self.weights_matrices = np.array(self.weights_matrices)
            self.bias_vectors = np.array(self.bias_vectors)
        else:
            data = np.load(structure)
            self.weights_matrices = data["w"]
            self.layers = len(self.weights_matrices) + 1
            self.bias_vectors = data["b"]
            self.structure = data["s"]

    def __str__(self):
        """
        :return: the weights and bias
        """

        visualize = "\nstructure of the network : {}\n".format(self.structure)
        for i in range(self.layers - 1):
            visualize = visualize + "\nweights of layer {0} and {1} : \n{2}\n" \
                                    "\nbias of layer {0} and {1} : \n{3}\n \n"\
                .format(i + 1, i + 2, self.weights_matrices[i], self.bias_vectors[i])
        visualize += "\n"

        return visualize


    def think(self, inputs):
        """
        method calculating the outputs from the inputs
        :param inputs = array of length(len(inputs_neuron))
        :return: array of length(len(outputs_neuron))
        """

        response_layer = sigmoid(self.weights_matrices[0] @ inputs.T + self.bias_vectors[0])
        for i in range(self.layers - 2):
            response_layer = sigmoid(self.weights_matrices[i + 1] @ response_layer.T + self.bias_vectors[i + 1])

        return response_layer


    def adjust(self, train_inputs, targets, learning_rate):
        """
        training method, needs to be ran multiples time
        :param train_inputs = array of length(len(inputs_neuron))
        :param targets = array of length(len(outputs_neuron))
        :param learning_rate = positive float
        :return: /
        """

        # compute the  inputs and outputs of each layer
        # initiate the inputs and the outputs
        response_layer = sigmoid(self.weights_matrices[0] @ train_inputs.T + self.bias_vectors[0])
        inputs = [train_inputs, response_layer]
        output_layer = [response_layer]

        for i in range(self.layers - 2):
            response_layer = sigmoid(self.weights_matrices[i + 1] @ response_layer.T + self.bias_vectors[i + 1])
            inputs.append(response_layer)  # the last append doesn't matter
            output_layer.append(response_layer)

        # compute the error of each layer
        layer_error = targets - output_layer[self.layers - 2]

        # adjusting the weights and bias by the delta's
        for i in range(self.layers - 1):  # va de gauche à droite, à changer vers de droite à gauche
            gradient = np.mat(learning_rate * layer_error.T * d_sigmoid(output_layer[self.layers - 2 - i]))  # gradient se met mal, .T?

            gradient_T = gradient.T
            mat_inputs = np.mat(inputs[self.layers - 2 - i])  # attention trop longue inputs
            delta_weights = gradient_T.dot(mat_inputs)

            self.weights_matrices[self.layers - 2 - i] += delta_weights
            self.bias_vectors[self.layers - 2 - i] += np.array(gradient)[0]

            layer_error = self.weights_matrices[self.layers - i - 2].T @ [layer_error]


    def train(self, train_inputs, targets, learning_rate, iterations, filename):
        """
        train loop and save the trained weights and bias to filename
        :param train_inputs = list of arrays of length(len(inputs_neuron))
        :param targets = list of arrays of length(len(outputs_neuron))
        :param learning_rate = positive float
        :param iterations = number of time we execute the loop
        :param filename =
        :return: the time of the process
        """

        t1 = time.time()
        t3 = time.time()
        for i in range(iterations):
            choice = np.random.random_integers(len(train_inputs)) - 1
            train_input = np.array(train_inputs)[choice]
            target = np.array(targets)[choice]

            self.adjust(train_input, target, learning_rate)
            t4 = time.time()
            if t4 - t3 > 300:  # save to filename avery 5 minutes
                self.save(filename)
                t3 = time.time()
        t2 = time.time()
        self.save(filename) # save at the end of the process

        return duration(t2 - t1)


    def save(self, filename):
        """
        save the weights and bias in the given file
        :param filename : name of the file where all the data will be saved
        :return: /
        """
        np.savez(filename, s=self.structure, w=self.weights_matrices, b=self.bias_vectors)


# functions
# useful functions for learning methods
def sigmoid(x):
    return 1/(1 + np.exp(-x))


def d_sigmoid(x):
    return x * (1 - x)


# optional function giving the time of the learning
def duration(d):
    h = 0
    while d >= 3600:
        h += 1
        d -= 3600
    m = 0
    while d >= 60:
        m += 1
        d -= 60
    d = int(d)
    return "{} hours, {} minutes et {} seconds".format(h, m, d)
