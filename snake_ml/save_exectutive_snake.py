from snake import SnakeGame
import numpy as np
import pygame
from réseaux.Multi_layer_NN import sorted_brains_scores
from matplotlib import pyplot as plt

# variables
n_generations = 40  # number of generations
n_batch = 100  # number of snakes in a batch
bests = 3  # number of best snakes that will be picked, not used atm
proportion = 0.5  # proportion in % of weights/biases that will be changed in the mutation
amplitude = 0.5  # maximum of change in a w/b that can occurs in the mutation
d_amplitude = 0.0005
init_moves = 60  # number of moves the snake can do, can increase with time?
add_moves = 100  # number of moves added when the snake eats food
screen = False  # see the screen or not
speed = 400  # speed in squares/s of the snake
size = 20  # size of the world
loaded = True  # if we want to evolve a saved snake
file = "impressive_snake.npz"  # file to load the snake to evolve
text = "cosinus snake"  # name of the graph
structure = [6]  # hidden_layers of the brain

# main program

if loaded:
    parents = file
else:
    parents = []

generations = np.empty(n_generations, dtype=bytearray)
scores_p = []
x = np.arange(n_generations)
y_scores = np.array([])
y_median = np.arange(n_generations)
for i in range(n_generations):
    children = np.empty(n_batch, dtype=tuple)

    for j in range(n_batch):
        pygame.init()
        children[j] = SnakeGame(parents, scores_p, structure,
                                proportion, amplitude,
                                init_moves, add_moves, i, j,
                                screen, speed, size, loaded).play
        pygame.quit()
    loaded = False
    ordered_children, scores = sorted_brains_scores(children)

    scores_p = np.trim_zeros(scores)
    parents = ordered_children[:np.size(scores_p)]
    print(scores_p)
    x_batch = np.ones(np.size(scores_p)) * i
    if len(scores_p) == 0:
        y_scores = np.append(y_scores, 0)
    else:
        y_scores = np.append(y_scores, scores_p[0])
    median_scores = np.sum(scores) / np.size(scores)
    y_median[i] = median_scores

    amplitude -= d_amplitude

    generations[i] = parents

plt.plot(x, y_scores, "b", label="best of gen")
plt.plot(x, y_median, "--b", label="median of scores", linewidth=2.0)
plt.suptitle(text, fontsize=32)
plt.xlabel("generation")
plt.ylabel("best of the generation")
plt.title("bests : {}   proportion : {}   amplitude : {}   n_batch : {}"
                "structure : {}   size_screen : {}   move_i : {}   move_add : {} "
                .format(bests, proportion, amplitude, n_batch, structure, size, init_moves, add_moves))
plt.show()

#generations[-1][0].save("meilleur_snake")
