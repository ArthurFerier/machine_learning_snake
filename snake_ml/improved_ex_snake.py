from snake_ml.improved_snake_cos import SnakeGame
import numpy as np
import pygame
from réseaux.Multi_layer_NN import sorted_brains_scores, duration
from matplotlib import pyplot as plt
import time
import os.path


# variables

# what I have to change possibly :
# test one snake on more than one game (can be bad luck for the snake else)


n_generations = 5              # number of generations
n_batch = 100                  # number of snakes in a batch
n_eval = 15                      # number of evaluations of the brain
bests = 1                       # number of best snakes that will be picked

proportion = 0.3                # proportion in % of weights/biases that will be changed in the mutation
d_proportion = 0                # decrease of the proportion for each bath
standard_dev = 2                   # standard deviation of the mutation
d_standard_dev = 0.1               # decrease of the amplitude for each generation

init_moves = 100                # number of moves the snake can do, can increase with time?
add_moves = 100                 # number of moves added when the snake eats food

screen = True                   # see the screen or not              !!!!!!!!!!! to implement
speed = 4000000                       # speed in squares/s of the snake
size = 17                      # size of the world

loaded = True                 # if we want to evolve a saved snake
file = "interesting_snakes/best_snake_score_50.npz"  # file to load the snake to evolve

save = True
text = "continuation best_snake_score_50"               # name of the graph
structure = [6, 6]            # hidden_layers of the brain
namefile = "continuation_best_snake_score_50"     # name of the file containing the graph of the results

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
                         proportion, standard_dev,
                         init_moves, add_moves, i,
                         screen, speed, size, loaded,
                         n_batch, n_eval).play
    pygame.quit()

    loaded = False
    ordered_children, scores = sorted_brains_scores(children, n_eval)
    print(scores)

    scores_p = scores[:bests]
    parents = ordered_children[:bests]
    print("{} : {}th generation".format(scores_p, i+1))
    x_batch = np.ones(np.size(scores_p)) * i
    if len(scores_p) == 0:
        y_scores = np.append(y_scores, 0)
    else:
        y_scores = np.append(y_scores, scores_p[0])
    median_scores = np.sum(scores) / np.size(scores)
    y_median[i] = median_scores

    generations[i] = parents
    ordered_children[0].save("best_of_gen/best_of_gen"+str(i))

    proportion -= d_proportion
    standard_dev -= d_standard_dev

t1 = time.perf_counter()
print(duration(t1 - t0))

plt.figure(figsize=(20, 15))
plt.plot(x, y_scores, "b", label="best of gen")
plt.plot(x, y_median, "--b", label="median of scores", linewidth=2.0)
plt.suptitle(text, fontsize=32)
plt.xlabel("generation")
plt.ylabel("best of the generation")
plt.title("bests : {}   proportion : {}   amplitude_max : {}   n_batch : {}  "
          "structure : {}   size_screen : {}   move_i : {}   move_add : {} "
          .format(bests, proportion, standard_dev, n_batch, structure, size, init_moves, add_moves))
number = 0
while os.path.exists("../data_snake_ml/{}_{}.png".format(namefile, number)):
    print("hello")
    number += 1

if save:
    plt.savefig("../data_snake_ml/{}_{}".format(namefile, number))
plt.show()

