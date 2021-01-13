import numpy as np
import time

#np.random.seed(2)
# main class
class MLNeuralNetwork:

    def __init__(self, structure):
        """
        initialise the network, with random weights and bias between [-1, 1]
        :param structure: list = [nbre_input, nbre_hidden1, nbre_hidden2, ..., nbre_output]
                          OR file to load weights and bias
                          OR list containing two parents neuralNetwork
        """

        if type(structure) == list:
            if type(structure[0]) == MLNeuralNetwork:  # cross-over
                # create empty neural network
                self.layers = len(structure[0].structure)
                self.structure = np.array(structure[0].structure)
                self.weights_matrices = []
                self.bias_vectors = []
                for i in range(self.layers - 1):
                    weight_matrix = np.empty((self.structure[i + 1], self.structure[i]))
                    bias_vector = np.empty(self.structure[i + 1])
                    self.weights_matrices.append(weight_matrix)
                    self.bias_vectors.append(bias_vector)
                # picking components of the parents to create the child
                # picking weights
                for e, matrix in enumerate(self.weights_matrices):
                    for f, line in enumerate(matrix):
                        for g in range(len(line)):
                            choice = int(np.random.choice(2, 1))
                            self.weights_matrices[e][f][g] = structure[choice].weights_matrices[e][f][g]
                # picking biases
                for e, vec in enumerate(self.bias_vectors):
                    for f in range(len(vec)):
                        choice = int(np.random.choice(2, 1))
                        self.bias_vectors[e][f] = structure[choice].bias_vectors[e][f]
                # converting to array for rapidity
                self.weights_matrices = np.array(self.weights_matrices, dtype=object)
                self.bias_vectors = np.array(self.bias_vectors, dtype=object)
            else:  # create random network
                self.layers = len(structure)
                self.structure = np.array(structure)
                self.weights_matrices = []
                self.bias_vectors = []
                for i in range(self.layers - 1):
                    weight_matrix = np.random.randn(structure[i+1], structure[i])
                    bias_vector = np.random.randn(structure[i + 1])
                    self.weights_matrices.append(weight_matrix)
                    self.bias_vectors.append(bias_vector)
                self.weights_matrices = np.array(self.weights_matrices, dtype="object")
                self.bias_vectors = np.array(self.bias_vectors, dtype="object")
        else:  # load network
            #np_load_old = np.load
            #np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)
            data = np.load(structure, allow_pickle=True)
            self.weights_matrices = data["w"]
            self.layers = len(self.weights_matrices) + 1
            self.bias_vectors = data["b"]
            self.structure = data["s"]

        self.score = 0

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
        visualize += "score : {}".format(self.score)

        return visualize


    def think(self, inputs):
        """
        method calculating the outputs from the inputs
        :param inputs = array of length(len(inputs_neuron))
        :return: array of length(len(outputs_neuron))
        """

        for w, b in zip(self.weights_matrices, self.bias_vectors):
            inputs = sigmoid(np.dot(w, inputs) + b)
        return sigmoid(inputs)

    def choice(self, array):
        choice = np.zeros(len(array))
        choice[np.argmax(array)] = 1
        return choice


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
            layers_error.append(layer_error)   # errors de droite à gauche

        # adjusting the weights and bias by the delta's
        size_outputs_error = len(layers_error) - 1
        for i in range(self.layers - 1):  # va de gauche à droite, à changer vers de droite à gauche
            gradient = learning_rate * layers_error[size_outputs_error - i] * d_sigmoid(output_layer[i])
            delta_weights = np.outer(gradient, inputs[i])

            self.weights_matrices[i] += delta_weights
            self.bias_vectors[i] += gradient


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
            if t4 - t3 > 300:  # save to filename every 5 minutes
                self.save(filename)
                t3 = time.time()
        t2 = time.time()
        self.save(filename)  # save at the end of the process

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

    def mutate(self, proportion, amplitude):
        """
        mutate the neural_network
        :param proportion: percentage ]0, 1] of the w and b that will be changed
        :param amplitude: maximum number added or subtracted to the w or b
        :return: /
        """
        for e, matrix in enumerate(self.weights_matrices):
            for f, line in enumerate(matrix):
                for g in range(len(line)):
                    if np.random.rand() < proportion:
                        self.weights_matrices[e][f][g] += (np.random.rand() * 2 - 1) * amplitude

        for e, vec in enumerate(self.bias_vectors):
            for f in range(len(vec)):
                if np.random.rand() < proportion:
                    self.bias_vectors[e][f] += (np.random.rand() * 2 - 1) * amplitude

    def mutate2(self, proportion, amplitude):
        for e, matrix in enumerate(self.weights_matrices):
            for f, line in enumerate(matrix):
                for g in range(len(line)):
                    if np.random.rand() < proportion:
                        self.weights_matrices[e][f][g] += np.random.randn()/5

                        if self.weights_matrices[e][f][g] > 1:
                            self.weights_matrices[e][f][g] = 1

                        if self.weights_matrices[e][f][g] < -1:
                            self.weights_matrices[e][f][g] = -1

        for e, vec in enumerate(self.bias_vectors):
            for f in range(len(vec)):
                if np.random.rand() < proportion:
                    self.bias_vectors[e][f] += np.random.randn()/5

                    if self.bias_vectors[e][f] > 1:
                        self.bias_vectors[e][f] = 1

                    if self.bias_vectors[e][f] < -1:
                        self.bias_vectors[e][f] = -1


# functions
# useful functions for learning methods
def sigmoid(x):
    return 1/(1 + np.exp(-x))


def d_sigmoid(x):
    return x * (1 - x)


# fonction to force a choice
def choice(array):
    choice = np.zeros(len(array))
    choice[np.argmax(array)] = 1
    return choice


# pooling functions

def sorted_brains_scores(brains, n_eval=1):
    """
    :param brains: array of type ndarray containing the brains
    :return: the array sorted with the biggest brain first
    """
    scores = get_scores(brains)/n_eval
    tri = np.argsort(scores)
    sorted_scores = np.sort(scores)[::-1]
    return brains[tri][::-1], sorted_scores


def pooling(sorted_brains, sorted_scores, consanguinity=True):  # !!!! scores est pas une variable locale???
    """
    choosing function
    :param sorted_brains: sorted array of brains
    :param sorted_scores : array of scores of the sorted brains
    :param consanguinity : boolean value letting the same brain reproduce itself or not
    (returning two times the same brain)
    :return: a couple of brains selected by their score in an array
    """
    # if brains and scores not sorted
    # sorted_brains, sorted_scores = sorted_brains_scores(brains)
    pool = np.copy(sorted_scores)
    if np.sum(pool) == 0 or np.sum(pool) == 1:
        return np.array([sorted_brains[0], sorted_brains[0]])

    # pool = squared difference between the scores
    pool -= int(pool[-1]**2/pool[0])
    pool **= 2
    # 36 25 16 9
    for i in range(len(pool) - 1):
        pool[-i - 2] += pool[-i - 1]
    sum = pool[0]
    # 86 50 25 9
    parents = np.empty(2, dtype=tuple)

    for j in range(2):  # 1 à changer
        choice = np.random.randint(sum + 1)  # between 0 and 86
        for i in range(np.size(pool) - 1):
            if choice >= pool[i + 1]:
                if j == 1 and consanguinity is not True and sorted_brains[i] == parents[0]:
                    parents[1] = sorted_brains[i+1]
                    break
                parents[j] = sorted_brains[i]
                break

        if parents[j] is None:
            if j == 1 and consanguinity is not True and sorted_brains[-1] == parents[0]:
                parents[1] = sorted_brains[-2]
            else:
                parents[j] = sorted_brains[-1]

    return np.array(parents)


def get_scores(brains):
    return np.vectorize(lambda obj: obj.score)(brains)


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
    return "{} hours, {} minutes and {} seconds".format(h, m, d)
