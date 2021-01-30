from improved_snake_cos import SnakeGame
from réseaux.Multi_layer_NN import MLNeuralNetwork
import pygame



pygame.init()


##### from just hte snakegame
#brain = MLNeuralNetwork("best_of_gen/bes_of_gen9")
#print(brain)
# parents, scores_p, structure,
#                  proportion, amplitude, moves, add_moves,
#                  generation, batch, screen, speed, size_world,
#                  loaded(mute the brain or not)
#a = SnakeGame("best_of_gen/best_of_gen9.npz",
#              [], [], 0, 0, 10000, 0, 0, 0, True, 15, 17, False).play


# try imprevod snake cos :
path = "best_of_gen/best_of_gen13.npz"  # best snake
#path = "best_of_gen/best_of_gen4.npz"

brain2 = MLNeuralNetwork(path)

a = SnakeGame(path, 0, [6, 6], 0, 0, 1000, 100, 1, True, 10, 17, False, 1).play

brain = MLNeuralNetwork(path)
print(brain)
print(brain2)
print(brain.compareTo(brain2))


pygame.quit()
