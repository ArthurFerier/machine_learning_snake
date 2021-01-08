from snake import SnakeGame
import pygame
from réseaux.Multi_layer_NN import *


pygame.init()
a = SnakeGame("D:\\Cloudstation\\Machine_learning\\snake_ml\\best_of_gen2\\best_of_gen8.npz",
              [], [], 0, 0, 10000, 0, 0, 0, True, 4, 20, True).play
print(a[0])
pygame.quit()
