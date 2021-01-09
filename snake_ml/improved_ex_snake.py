from snake_ml.improved_snake_cos import SnakeGame
import numpy as np
import pygame
from réseaux.Multi_layer_NN import sorted_brains_scores, duration
from matplotlib import pyplot as plt
import time


# variables
n_generations = 20              # number of generations
n_batch = 500                  # number of snakes in a batch
n_eval = 1                      # number of evaluations of the brain
bests = 1                       # number of best snakes that will be picked
proportion = 0.5                # proportion in % of weights/biases that will be changed in the mutation
d_proportion = 0                # decrease of the proportion for each bath
amplitude = 3                   # maximum of change in a w/b that can occurs in the mutation
d_amplitude = 0.1               # decrease of the amplitude for each batch
init_moves = 200                # number of moves the snake can do, can increase with time?
add_moves = 100                 # number of moves added when the snake eats food
screen = True                   # see the screen or not
bool_speed = False              # tame the speed of the snake or not
speed = 1000                       # speed in squares/s of the snake
size = 20                      # size of the world
loaded = False                  # if we want to evolve a saved snake
file = "best_of_gen/best_of_gen11.npz"     # file to load the snake to evolve
text = "24_snake"               # name of the graph
structure = [16, 16]            # hidden_layers of the brain
namefile = "first_24_snake"     # name of the file containing the graph of the results

# main program

t0 = time.perf_counter()
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

    pygame.init()
    children = SnakeGame(parents, scores_p, structure,
                         proportion, amplitude,
                         init_moves, add_moves, i,
                         screen, speed, size, loaded,
                         n_batch).play
    pygame.quit()
    loaded = False
    ordered_children, scores = sorted_brains_scores(children)

    scores_p = scores[:bests]
    parents = ordered_children[:bests]
    print("{} : {}th generation".format(scores_p, i))
    x_batch = np.ones(np.size(scores_p)) * i
    if len(scores_p) == 0:
        y_scores = np.append(y_scores, 0)
    else:
        y_scores = np.append(y_scores, scores_p[0])
    median_scores = np.sum(scores) / np.size(scores)
    y_median[i] = median_scores

    generations[i] = parents
    ordered_children[0].save("best_of_gen2/best_of_gen"+str(i))

    proportion -= d_proportion
    amplitude -= d_amplitude

t1 = time.perf_counter()
print(duration(t1 - t0))

plt.plot(x, y_scores, "b", label="best of gen")
plt.plot(x, y_median, "--b", label="median of scores", linewidth=2.0)
plt.suptitle(text, fontsize=32)
plt.xlabel("generation")
plt.ylabel("best of the generation")
plt.title("bests : {}   proportion : {}   amplitude : {}   n_batch : {}  "
          "structure : {}   size_screen : {}   move_i : {}   move_add : {} "
          .format(bests, proportion, amplitude, n_batch, structure, size, init_moves, add_moves))
plt.savefig("data_snake_ml/{}".format(namefile))
plt.show()

