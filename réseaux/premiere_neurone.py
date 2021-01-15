# first try of a brain, with only one neuron


from numpy import random
import sympy
import time

#  training set:
#  si input2 == 1 : return 1
#  list(input1, input2, input 3, output)
training_set = [[1, 1, 0, 1], [1, 0, 0, 0], [0, 0, 1, 0], [1, 1, 1, 1], [0, 0, 0, 0]]


class Neuron:

    def __init__(self):
        self.w1 = random.random()
        self.w2 = random.random()
        self.w3 = random.random()
        self.b = random.random()

    def response(self, input1, input2, input3):
        #  réponse aux inputs fournis
        return self.w1 * input1 + self.w2 * input2 + self.w3 * input3 + self.b

    def response_train(self, w, x, y, z, input1, input2, input3):
        return w * input1 + x * input2 + y * input3 + z

    def sigmoide(self, x):
        #  fonction sigmoïde
        return 1 / (1 + sympy.exp(-x))

    def error_sq(self, prediction, output):
        return (prediction - output) ** 2

    def prediction(self, input1, input2, input3):
        return self.sigmoide(self.response(input1, input2, input3))

    def train(self, training_set, iterations, training_rate):
        w = sympy.Symbol("w")
        x = sympy.Symbol("x")
        y = sympy.Symbol("y")
        z = sympy.Symbol("z")

        for i in range(iterations):
            ri = random.randint(len(training_set))
            point = training_set[ri]

            equation = self.sigmoide(self.response_train(w, x, y, z, point[0], point[1], point[2]))
            cost = (equation - point[3]) ** 2

            d_costw1 = cost.diff(w)
            d_costw2 = cost.diff(x)
            d_costw3 = cost.diff(y)
            d_costb = cost.diff(z)

            self.w1 = self.w1 - training_rate * d_costw1.subs([(w, self.w1), (x, self.w2), (y, self.w3), (z, self.b)])

            self.w2 = self.w2 - training_rate * d_costw2.subs([(w, self.w1), (x, self.w2), (y, self.w3), (z, self.b)])

            self.w3 = self.w3 - training_rate * d_costw3.subs([(w, self.w1), (x, self.w2), (y, self.w3), (z, self.b)])


            self.b = self.b - training_rate * d_costb.subs([(w, self.w1), (x, self.w2), (y, self.w3), (z, self.b)])



def temps(d):
    h = 0
    while d >= 3600:
        h += 1
        d -= 3600
    m = 0
    while d >= 60:
        m += 1
        d -= 60
    d = int(d)
    return "{} heures, {} minutes et {} secondes".format(h, m, d)


a = Neuron()
t1 = time.time()
a.train(training_set, 1000, 0.1)
t2 = time.time()
print(a.prediction(1, 0, 0))  # retourne 0 norm
print(a.prediction(0, 1, 0))  # retourne 1 norm => cet exemple ne vient pas du training set
print("temps d'apprentissage : {}".format(temps(t2 - t1)))


