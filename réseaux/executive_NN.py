from Multi_layer_NN import *

brain = MLNeuralNetwork([2, 2, 1])

#print(brain.think(np.ones(2)))


inputs = [[0, 0], [1, 1]]

targets = [0, 1]


brain.train(inputs, targets, 0.6, 100000, "XOR_problem")


print(brain.think(np.array([0, 1]))[0])  # 1
print(brain.think(np.array([1, 1]))[0])  # 0
print(brain.think(np.array([0, 0]))[0])  # 0
print(brain.think(np.array([1, 0]))[0])  # 1

#print(brain)
