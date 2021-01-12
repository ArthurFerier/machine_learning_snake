from snake import SnakeGame
import pygame



pygame.init()
SnakeGame("best_multiple_tries/best_of_gen19.npz", [], [], 0, 0, 10000, 0, 0, 0, True, 4, 20, True).play
pygame.quit()
