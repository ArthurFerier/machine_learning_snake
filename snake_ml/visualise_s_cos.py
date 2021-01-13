from snake import SnakeGame
from réseaux.Multi_layer_NN import MLNeuralNetwork
import pygame



pygame.init()

brain = MLNeuralNetwork("interesting_snakes/impressive_self_avoiding.npz")
print(brain)
a = SnakeGame("best_of_gen/best_of_gen0.npz",
              [], [], 0, 0, 10000, 0, 0, 0, True, 15, 20, False).play

pygame.quit()
