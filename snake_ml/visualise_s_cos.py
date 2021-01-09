from snake import SnakeGame
import pygame



pygame.init()
SnakeGame("best_of_gen2/best_of_gen8.npz", [], [], 0, 0, 10000, 0, 0, 0, True, 4, 20, True).play
pygame.quit()
