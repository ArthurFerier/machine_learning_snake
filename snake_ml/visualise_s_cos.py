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
brain = MLNeuralNetwork("/home/marty/informatique/machine_learning/snake_ml/interesting_snakes/best_snake_score_50.npz")
print(brain)
a = SnakeGame("interesting_snakes/best_snake_score_50.npz", 0, [6, 6], 0, 0, 1000, 100, 1, True, 2, 17, False, 1).play

brain = MLNeuralNetwork("/home/marty/informatique/machine_learning/snake_ml/interesting_snakes/best_snake_score_50.npz")
print(brain)

pygame.quit()
