import numpy as np


def sigmoid(x):
    return 1/(1 + np.exp(-x))


def d_sigmoid(x):
    return x * (1 - x)


class NeuralNetwork:
    def __init__(self, structure):
        """
        initialise the network, with random weights between [-1, 1]
        :param structure: list = [nbre_input, nbre_hidden, nbre_output]
        """

        self.matrix_IH = (np.random.rand(structure[1], structure[0])) * 2 - 1
        self.matrix_HO = (np.random.rand(structure[2], structure[1])) * 2 - 1
        self.hidden_bias = (np.random.random_sample(structure[1])) * 2 - 1
        self.output_bias = (np.random.random_sample(structure[2])) * 2 - 1


    def feed_forward(self, inputs):
        """
        method calculating the outputs from the inputs
        :param inputs = array of length(len(inputs_neuron))
        :return: array of length(len(outputs_neuron))
        """
        return self.outputs(self.hidden_outputs(inputs))

    def hidden_outputs(self, inputs):
        """
        calculate the output of the hidden layer
        :param inputs = array of length(len(inputs_neuron))
        :return: array of length(len(hidden_neurons))
        """
        return sigmoid(self.matrix_IH @ inputs.T + self.hidden_bias)

    def outputs(self, hidden_outputs):
        """
        calculate the output from the output layer
        :param hidden_outputs = array of length(len(hidden_neurons))
        :return: array of length(len(outputs_neuron))
        """
        return sigmoid(self.matrix_HO @ hidden_outputs.T + self.output_bias)


    def adjust(self, train_inputs, targets, learning_rate):
        """
        training method, needs to be ran multiples time
        :param train_inputs = array of length(len(inputs_neuron))
        :param targets = array of length(len(outputs_neuron))
        :param learning_rate = positive float
        :return: nothing, adjust the weights and the biases of the NN
        """

        # calculate the outputs and the hidden_outputs
        hidden_outputs = self.hidden_outputs(train_inputs)
        outputs = self.outputs(hidden_outputs)
        # calculate the output_errors
        output_errors = targets - outputs
        # calculate the hidden_errors
        hidden_errors = self.matrix_HO.T @ output_errors


        # calculate the changes of the inputs weights
        input_gradient = np.mat(learning_rate * hidden_errors * d_sigmoid(hidden_outputs))
        input_gradient_T = input_gradient.T
        train_inputs = np.mat(train_inputs)
        delta_hid_weights = input_gradient_T.dot(train_inputs)
        # adjusting the hidden weights
        self.matrix_IH += delta_hid_weights
        # adjusting the hidden biases
        self.hidden_bias += np.array(input_gradient)[0]


        # calculate the changes of the outputs weights
        output_gradient = np.mat(d_sigmoid(outputs) * output_errors * learning_rate)
        output_gradient_T = output_gradient.T
        hidden_outputs_mat = np.mat(hidden_outputs)
        delta_out_weights = output_gradient_T.dot(hidden_outputs_mat)

        # adjusting the output weights
        self.matrix_HO += delta_out_weights
        # adjusting the output biases
        self.output_bias += np.array(output_gradient)[0]

        return

    def train(self, train_inputs, targets, learning_rate, iterations):
        """
        train loop
        :param train_inputs = list of arrays of length(len(inputs_neuron))
        :param targets = list of arrays of length(len(outputs_neuron))
        :param learning_rate = positive float
        :param iterations = number of time we execute the loop
        :return: nothing, adjust the weights and the biases of the NN
        """

        for i in range(iterations):
            choice = np.random.random_integers(len(train_inputs)) - 1
            train_input = np.array(train_inputs)[choice]
            target = np.array(targets)[choice]

            self.adjust(train_input, target, learning_rate)

