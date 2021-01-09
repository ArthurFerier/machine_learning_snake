import numpy as np
import time

oui = "oui"
# np.random.seed(1)


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
        :return: the structure of hte NN, the weights and bias
        """

        visualize = "\nstructure of the network : {}\n".format(self.structure)
        for i in range(self.layers - 1):
            visualize = visualize + "\nweights of layer {0} and {1} : \n{2}\n" \
                                    "\nbias of layer {1} : \n{3}\n \n"\
                .format(i + 1, i + 2, self.weights_matrices[i], self.bias_vectors[i])
        visualize += "\n"

        return visualize


    def think(self, inputs):
        """
        method calculating the outputs from the inputs
        :param inputs = array of length(len(inputs_neuron))
        :return: array of length(len(outputs_neuron))
        """

        for w, b in zip(self.weights_matrices, self.bias_vectors):
            inputs = sigmoid(np.dot(w, inputs) + b)
        return inputs


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
        inputs = [train_inputs]
        output_layer = []

        for w, b in zip(self.weights_matrices, self.bias_vectors):
            train_inputs = sigmoid(w @ train_inputs.T + b)
            inputs.append(train_inputs)  # the last append doesn't matter
            output_layer.append(train_inputs)

        # compute the error of each layer
        layer_error = targets - output_layer[self.layers - 2]
        layers_error = [layer_error]

        for i in range(self.layers - 2):  # va de droite à gauche
            layer_error = self.weights_matrices[self.layers - i - 2].T @ layers_error[i]
            layers_error.append(layer_error)   # errors de droite àgauche

        # adjusting the weights and bias by the delta's
        size_outputs_error = len(layers_error) - 1
        for i in range(self.layers - 1):  # va de gauche à droite, à changer vers de droite à gauche
            gradient = np.mat(learning_rate * layers_error[size_outputs_error - i] * d_sigmoid(output_layer[i]))
            gradient_T = gradient.T
            mat_inputs = np.mat(inputs[i])
            delta_weights = gradient_T.dot(mat_inputs)

            self.weights_matrices[i] += delta_weights
            self.bias_vectors[i] += np.array(gradient)[0]


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

    def change(self, changes):
        """

        :param changes: list containing all the changes we want to apply,
        the elements are : [place, new_value], place is :
        [ w or b, 0: weights of layer 1-2 or bias of layer 2,
        line of the matrix, column of the matrix (not needed if bias)]
        :return: /
        """
        for data in changes:
            place = data[0]
            if place[0] == "w":
                self.weights_matrices[place[1]][place[2]][place[3]] = data[1]
            else:
                self.bias_vectors[place[1]][place[2]] = data[1]




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
