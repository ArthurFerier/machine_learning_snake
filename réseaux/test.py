import numpy as np
from réseaux.Multi_layer_NN import *
from matplotlib import pyplot as plt

print(np.zeros(1))

"""
a = np.array([MLNeuralNetwork([2, 2, 1]), MLNeuralNetwork([2, 3, 1]), MLNeuralNetwork([2, 4, 1])])

a[0].score = 0
a[1].score = 0
a[2].score = 0

print(pooling(a))

count1 = 0
count2 = 0
count3 = 0

for i in range(100):
    num = pooling(a)
    if num == a[0]:
        count1 += 1

    elif num == a[1]:
        count2 += 1

    elif num == a[2]:
        count3 += 1

print(count1)
print(count2)
print(count3)
"""