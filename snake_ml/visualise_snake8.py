from snake_8 import SnakeGame
import pygame
from réseaux.Multi_layer_NN import *


pygame.init()
a = SnakeGame("D:\\Cloudstation\\Machine_learning\\snake_ml\\meilleur_snake.npz",
              [], [], 0, 0, 10000, 0, 0, True, 4, 20, True, 1, True, 1).play
print(a[0])
pygame.quit()


