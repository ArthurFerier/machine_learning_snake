from snake import SnakeGame
from réseaux.Multi_layer_NN import MLNeuralNetwork
import pygame



pygame.init()

#brain = MLNeuralNetwork("best_multiple_tries/best_of_gen7.npz")
#print(brain)
a = SnakeGame("best_multiple_tries/best_of_gen7.npz", [], [], 0, 0, 10000, 0, 0, 0, True, 4, 17, True).play

pygame.quit()
