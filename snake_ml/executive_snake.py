from snake import SnakeGame
import numpy as np
import pygame
from réseaux.Multi_layer_NN import sorted_brains_scores, duration
from matplotlib import pyplot as plt
import time

# variables
n_generations = 20            # number of generations
n_batch = 200                # number of snakes in a batch
bests = 6                    # number of best snakes that will be picked, not used atm
proportion = 0.05            # proportion in % of weights/biases that will be changed in the mutation
amplitude = 2                # maximum of change in a w/b that can occurs in the mutation
init_moves = 100              # number of moves the snake can do, can increase with time?
add_moves = 120              # number of moves added when the snake eats food
screen = True               # see the screen or not
speed = 3                   # speed in squares/s of the snake
size = 30                    # size of the world
loaded = False               # if we want to evolve a saved snake
file = "impressive_snake.npz"  # file to load the snake to evolve
text = "cosinus snake"       # name of the graph
structure = [12, 6, 4]           # hidden_layers of the brain


# main program

t1 = time.perf_counter()
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
#colors = ['b', 'g']
for color in colors:
    generations = np.empty(n_generations, dtype=bytearray)
    scores_p = []
    x = np.arange(n_generations)
    y_scores = np.array([])
    if loaded:
        parents = file
    else:
        parents = []
    y_median = np.empty(n_generations)
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
        median_scores = np.sum(scores)/np.size(scores)
        print(median_scores)
        y_median[i] = median_scores

        scores_p = np.trim_zeros(scores)
        parents = ordered_children[:np.size(scores_p)]
        print(scores_p)
        if len(scores_p) == 0:
            y_scores = np.append(y_scores, 0)
        else:
            y_scores = np.append(y_scores, scores_p[0])

        generations[i] = parents  # ! à créer le nouveau batch

    plt.plot(x, y_scores, color, label="best of gen")
    plt.plot(x, y_median, "--"+color, label="median of scores", linewidth=2.0)
t2 = time.perf_counter()
print(duration(t2 - t1))

plt.suptitle(text, fontsize=32)
plt.xlabel("generation")
plt.ylabel("best of the generation")
plt.title("bests : {}   proportion : {}   amplitude : {}   n_batch : {}"
                "structure : {}   size_screen : {}   move_i : {}   move_add : {} "
                .format(bests, proportion, amplitude, n_batch, structure, size, init_moves, add_moves))
plt.show()

#generations[-1][0].save("meilleur_snake")
