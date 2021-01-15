# machine_learning_snake

## Description of the project

The aim of this project is to develop my own neural network library in python
using only numpy as external library.
When it was done, I had to test it and chose to train neural networks on the snake 
game. I can't find the site where I initially downloaded the game. 

The graphs of the results of all the training are in the folder data_snake_ml.
They aren't really organised but that's what I used.

The rest is in the folder snake_ml.
In it, there are sub-folders with saves of snakes. 
The snakes are saved in .npz format by my save method of the library I created, 
which is a format file used by numpy.

The folder interesting_snakes contains the best snakes I trained so far or snakes that 
behaves surprisingly.

The folder best_of_gen and best_tries_old are folder I used to store every generation 
of the running learning_algorithm.    
The best snakes of each generations are currently stored in the folder best_of_gen.

The rest of the files in the folder snake_ml are all the python scripts to run the snake.

I trained two types of snakes :
- snake_8
- snake_cos

They differ in the input they take to feed their neural networks.  
The snake 8 takes 16 inputs that are classified in 2 categories : 
the food and the obstacles.
The snake sees in 8 directions for each, above, under, right, left, above_right, above_left, ...
It was my first try of giving inputs to the snake and after seeing it again
two years after the beginning of the project made me realise that they are not correct :-)
This snake is not my proudest realisation, and I didn't go far with it.

The second snake is the Snake_cos. It takes 10 inputs.  
8 from the 8 cardinal points of the distance of the obstacles.  
1 to know the direction the head points on.  
1 that gives the cosinus value of the food with respect to the position of the head of the snake,
and the direction that the snake points on.
The brain has 3 output nodes : MOVE_LEFT, MOVE_RIGHT, CONTINUE.
I achieved the best performances with this snake.

## How to use it

### what to copy from the repository

This repository is the backup of my entire project. All the folders are not worth coping.
The only directories worth it are réseaux and snake_ml.
I made this project on A linux laptop. 
There are some part of the code that uses pathes to load or save some files.
These path are probably no more correct if you use a Windows or macOS.


In summary, copy the folders réseaux and snake_ml and put them into a main folder.
Open the main folder with a Python environment and you can use it 
after installing all the librairies needed !

### evolve a snake

To evolve a cos_snake from nothing. Go to the file improved_ex_snake.py.  
At the beginning of the file, there is a bunch of parameters to set to manage the evolution.
They are all commented, there shouldn't be any problems to understand them (I hope so).
Run this file, each generation will be daved in the best_of_gen directory. 

When the script is done, you can visualise each saved snake with the script visualise_s_cos.py and enjoy !

