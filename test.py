import numpy as np
from réseaux.Multi_layer_NN import *

brains = np.array([MLNeuralNetwork([2, 2, 1]), MLNeuralNetwork([2, 3, 1]), MLNeuralNetwork([2, 4, 1]), MLNeuralNetwork([2, 5, 1])])
sc = scores(brains)
print(sc)


"""
brains = np.array([MLNeuralNetwork([2, 2, 1]), MLNeuralNetwork([2, 3, 1]), MLNeuralNetwork([2, 4, 1]), MLNeuralNetwork([2, 5, 1])])
brains[0].score = 7
brains[1].score = 13
brains[2].score = 4
brains[3].score = 20

count1 = 0
count2 = 0
count3 = 0
count4 = 0

for i in range(1000):
    a = pooling(brains)
    
    if a == brains[0]:
        count1 += 1
    
    if a == brains[1]:
        count2 += 1
    
    if a == brains[2]:
        count3 += 1
    
    if a == brains[3]:
        count4 += 1

print(count1)
print(count2)
print(count3)
print(count4)
    
"""

"""
oui = np.array(["a", "b", "c", "d"])
non = np.array([100, 99, 98, 97])

counta = 0
countb = 0
countc = 0
countd = 0


for i in range(1000):
    merde = pooling(oui, non)
    if merde == "a":
        counta += 1
        
    elif merde == "b":
        countb += 1
        
    elif merde == "c":
        countc += 1
        
    elif merde == "d":
        countd += 1


print(counta)
print(countb)
print(countc)
print(countd)
    
    

"""