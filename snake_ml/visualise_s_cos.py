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
path = "interesting_snakes/best_snake_score_50.npz"
brain = MLNeuralNetwork(path)
#print(brain)
a = SnakeGame(path, 0, [6, 6], 0, 0, 100000, 10000, 1, True, 3, 25, False, 1, False, False, n_eval=1).play

# moyenne pour le snake avec le sonar
"""
66
70
37
71
73
72
78
84
66
49
53
74
62
96
71
59
86
98
71
51
moyenne : 69.35
"""

"""
snake sans sonar
64
39
49
22
69
69
58
57
76
62
57
22
24
80
56
41
47
41
47
83
===> moyenne : 49.7
"""


pygame.quit()
