# on fait les modifs ici pour le machine learning
from collections import deque

# import numpy as np
import pygame
from random import randrange
from pygame.locals import *
from réseaux.Multi_layer_NN import *
import time


class Vector(tuple):
    """A tuple that supports some vector operations.

    v, w = Vector((1, 2)), Vector((3, 4))
    v + w, w - v, v * 10, 100 * v, -v
    ((4, 6), (2, 2), (10, 20), (100, 200), (-1, -2))
    """

    def __add__(self, other):
        return Vector(v + w for v, w in zip(self, other))

    def __radd__(self, other):
        return Vector(w + v for v, w in zip(self, other))

    def __sub__(self, other):
        return Vector(v - w for v, w in zip(self, other))

    def __rsub__(self, other):
        return Vector(w - v for v, w in zip(self, other))

    def __mul__(self, s):
        return Vector(v * s for v in self)

    def __rmul__(self, s):
        return Vector(v * s for v in self)

    def __neg__(self):
        return -1 * self



FPS = 60  # Game frames per second
SEGMENT_SCORE = 1  # Score per segment

SNAKE_SPEED_INCREMENT = 0.25  # Snake speeds up this much each time it grows
SNAKE_START_LENGTH = 5  # Initial snake length in segments

WORLD_SIZE = Vector((20, 20))  # World size, in blocks
BLOCK_SIZE = 24  # Block size, in pixels

BACKGROUND_COLOR = 0, 0, 0
SNAKE_COLOR = 0, 255, 0
SNAKE_COLOR2 = 255, 255, 0
SNAKE_COLOR3 = [255, 0, 100]
FOOD_COLOR = 255, 0, 0
DEATH_COLOR = 255, 0, 0
TEXT_COLOR = 0, 0, 255

DIRECTION_UP = Vector((0, -1))
DIRECTION_DOWN = Vector((0, 1))
DIRECTION_LEFT = Vector((-1, 0))
DIRECTION_RIGHT = Vector((1, 0))
DIRECTION_DR = DIRECTION_DOWN + DIRECTION_RIGHT

# Map from PyGame key event to the corresponding direction.
KEY_DIRECTION = {
    K_q: DIRECTION_UP, K_UP: DIRECTION_UP,
    K_s: DIRECTION_DOWN, K_DOWN: DIRECTION_DOWN,
    K_a: DIRECTION_LEFT, K_LEFT: DIRECTION_LEFT,
    K_d: DIRECTION_RIGHT, K_RIGHT: DIRECTION_RIGHT,
}

time_passes = True
de_infinite = False
tik = 1
free_block_above = []
free_block_left = []
free_block_right = []
block_encountered_left = []
block_encountered_right = []
block_encountered_above = []

class Snake(object):
    def __init__(self, start, start_length, pot_parents, scores_p, proportion, amplitude, batch, speed, loaded, struct):
        self.speed = speed  # Speed in squares per second.
        self.timer = 1.0 / self.speed  # Time remaining to next movement.
        self.growth_pending = 0  # Number of segments still to grow.
        self.direction = DIRECTION_UP  # Current movement direction.
        self.segments = deque([start - self.direction * i for i in range(start_length)])
        if type(pot_parents) == str and loaded:
            self.brain = MLNeuralNetwork(pot_parents)
            self.brain.mutate(proportion, amplitude)
        elif type(pot_parents) == str and not loaded:
            self.brain = MLNeuralNetwork(pot_parents)
        elif type(pot_parents) == MLNeuralNetwork:
            self.brain = pot_parents
            self.brain.mutate(proportion, amplitude)
        else:
            if len(pot_parents) == 0:
                structure = np.concatenate(([7], struct, [3])).tolist()
                self.brain = MLNeuralNetwork(structure)
            else:
                parents = pooling(pot_parents, scores_p).tolist()
                self.brain = MLNeuralNetwork(parents)
                self.brain.mutate(proportion, amplitude)

    def __iter__(self):
        return iter(self.segments)

    def __len__(self):
        return len(self.segments)

    def head(self):
        """Return the position of the snake's head."""
        return self.segments[0]

    def update(self, dt, direction):
        """Update the snake by dt seconds and possibly set direction."""
        self.timer -= dt
        if self.timer > 0:
            # Nothing to do yet.
            return


        if self.direction == DIRECTION_UP:
            if direction == "middle":
                self.direction = DIRECTION_UP
            elif direction == "left":
                self.direction = DIRECTION_LEFT
            elif direction == "right":
                self.direction = DIRECTION_RIGHT

        elif self.direction == DIRECTION_DOWN:
            if direction == "middle":
                self.direction = DIRECTION_DOWN
            elif direction == "left":
                self.direction = DIRECTION_RIGHT
            elif direction == "right":
                self.direction = DIRECTION_LEFT

        elif self.direction == DIRECTION_RIGHT:
            if direction == "middle":
                self.direction = DIRECTION_RIGHT
            elif direction == "left":
                self.direction = DIRECTION_UP
            elif direction == "right":
                self.direction = DIRECTION_DOWN

        elif self.direction == DIRECTION_LEFT:
            if direction == "middle":
                self.direction = DIRECTION_LEFT
            elif direction == "left":
                self.direction = DIRECTION_DOWN
            elif direction == "right":
                self.direction = DIRECTION_UP

        self.timer += 1 / self.speed
        # Add a new head.
        self.segments.appendleft(self.head() + self.direction)
        if self.growth_pending > 0:
            self.growth_pending -= 1
        else:
            # Remove tail.
            self.segments.pop()

    def grow(self):
        """Grow snake by one segment and speed up."""
        self.growth_pending += 1
        self.speed += SNAKE_SPEED_INCREMENT
        self.speed = 20

    def self_intersecting(self):
        """Is the snake currently self-intersecting?"""
        it = iter(self)
        head = next(it)
        return head in it


class SnakeGame(object):
    def __init__(self, parents, scores_p, structure,
                 proportion, amplitude, moves, add_moves,
                 generation, screen, speed, size, loaded, n_batch, sonar, visualising_sonar, n_eval=1):
        pygame.display.set_caption('PyGame Snake')
        self.block_size = BLOCK_SIZE
        self.see = screen
        if self.see:
            self.window = pygame.display.set_mode(
                Vector((size, size)) * self.block_size)  # turn off if doesn't want to see the screen
        self.screen = pygame.display.get_surface()
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font('freesansbold.ttf', 20)
        self.world = Rect((0, 0), Vector((size, size)))
        self.size = size
        self.reset(parents, scores_p, proportion, amplitude, 0, speed, loaded, structure, same=False)
        self.moves = moves
        self.add_moves = add_moves
        self.generation = generation
        self.batch = 0
        self.brains = np.empty(n_batch, dtype=tuple)
        self.n_batch = n_batch

        self.parents = parents
        self.scores_p = scores_p
        self.proportion = proportion
        self.amplitude = amplitude
        self.speed = speed
        self.loaded = loaded
        self.structure = structure
        self.save_moves = moves
        self.n_eval = n_eval
        self.visualising_sonar = visualising_sonar
        self.use_sonar = sonar


    def reset(self, parents, scores_p, proportion, amplitude, batch, speed, loaded, structure, same=True):
        """Start a new game."""
        self.playing = True
        self.next_direction = DIRECTION_UP
        self.score = 0
        if not same:
            self.snake = Snake(self.world.center, SNAKE_START_LENGTH,
                               parents, scores_p, proportion,
                               amplitude, batch, speed, loaded, structure)
        else:
            self.snake.speed = speed
            self.snake.direction = DIRECTION_UP
            self.snake.segments = deque([self.world.center - DIRECTION_UP * i for i in range(SNAKE_START_LENGTH)])
        # à modifier en boucle for
        self.food = set()
        self.add_food()

    def add_food(self):
        """Ensure that there is at least one piece of food.
        """
        while not (self.food and randrange(4)):
            global food
            food = Vector(map(randrange, self.world.bottomright))
            # todo : remettre la proba que la food soit contre un mur pour le training
            if food not in self.food \
                    and food not in self.snake \
                    and food[0] != self.world.center[0]\
                    and food[0] != 0 \
                    and food[0] != self.size - 1 \
                    and food[1] != 0 \
                    and food[1] != self.size - 1:
                self.food.add(food)
                break

    def input(self, e):
        """Process keyboard event e."""
        if e.key in KEY_DIRECTION:
            self.next_direction = KEY_DIRECTION[e.key]
        if e.key == K_SPACE:
            print("paused")
            global time_passes
            time_passes = not time_passes

    def brain_action(self, actions):
        if actions[0] == 1:
            self.next_direction = "left"

        if actions[1] == 1:
            self.next_direction = "middle"

        if actions[2] == 1:
            self.next_direction = "right"


    def update(self, dt):
        """Update the game by dt seconds."""
        self.snake.update(dt, self.next_direction)

        # If snake hits a food block, then consume the food, add new
        # food and grow the snake.
        head = self.snake.head()
        if head in self.food:
            self.food.remove(head)
            self.add_food()
            self.snake.grow()
            self.score += 1  # len(self.snake) * SEGMENT_SCORE
            self.moves += self.add_moves

        # If snake collides with self or the screen boundaries, then
        # it's game over.
        if self.snake.self_intersecting() or not self.world.collidepoint(self.snake.head()):
            self.playing = False

    def block(self, p):
        """Return the screen rectangle corresponding to the position p."""
        return Rect(p * self.block_size, DIRECTION_DR * self.block_size)

    def draw_text(self, text, p):
        """Draw text at position p."""
        self.screen.blit(self.font.render(text, True, TEXT_COLOR), p)

    def draw(self, eval=0):
        """Draw game (while playing)."""
        self.screen.fill(BACKGROUND_COLOR)

        snake_length = len(self.snake)
        for p in self.snake:
            pygame.draw.rect(self.screen, SNAKE_COLOR3, self.block(p))
            SNAKE_COLOR3[0] -= int(255/snake_length)

        SNAKE_COLOR3[0] = 255

        for f in self.food:
            pygame.draw.rect(self.screen, FOOD_COLOR, self.block(f))
        self.draw_text("Score: {}".format(self.score), (20, 20))
        self.draw_text("generation: {}".format(self.generation), (20, 40))
        self.draw_text("batch: {}".format(self.batch), (20, 60))
        self.draw_text("essai {} of brain".format(eval), (20, 80))

    def draw_sonar(self, eval, cubes_diff_light):
        """Draw game (while playing)."""
        self.screen.fill(BACKGROUND_COLOR)

        snake_length = len(self.snake)
        # i can print this rectangle
        for p in self.snake:
            pygame.draw.rect(self.screen, SNAKE_COLOR3, self.block(p))
            SNAKE_COLOR3[0] -= int(255/snake_length)

        SNAKE_COLOR3[0] = 255

        for f in self.food:
            pygame.draw.rect(self.screen, FOOD_COLOR, self.block(f))
        self.draw_text("Score: {}".format(self.score), (20, 20))
        self.draw_text("generation: {}".format(self.generation), (20, 40))
        self.draw_text("batch: {}".format(self.batch), (20, 60))
        self.draw_text("essai {} of brain".format(eval), (20, 80))

        # drawing the fun part

        # array of the brightness of the yellow
        n_brightnesses = int((len(cubes_diff_light)+1)/2)
        adding_brightness = int(180 / n_brightnesses)
        brightnesses = np.empty(len(cubes_diff_light))
        for i in range(n_brightnesses):
            brightnesses[i] = (i+1)*adding_brightness
        for i in range(n_brightnesses-1):
            brightnesses[n_brightnesses+i] = adding_brightness*n_brightnesses - (i+1)*adding_brightness

        # funzzz
        for i in range(len(cubes_diff_light)):
            for blocks in cubes_diff_light[i]:
                pygame.draw.rect(self.screen, (int(brightnesses[i]), int(brightnesses[i]), 0),
                                 Rect((blocks[0]*self.block_size, blocks[1]*self.block_size),
                                       DIRECTION_DR * self.block_size))

        pygame.display.flip()


    def draw_death(self):
        """Draw game (after game over)."""
        self.screen.fill(DEATH_COLOR)
        self.draw_text("Game over! Press Space to start a new game", (20, 150))
        self.draw_text("Your score is: {}".format(self.score), (140, 180))


    def part_right(self, x, y):
        x_head = self.snake.segments[0][0]
        y_head = self.snake.segments[0][1]
        if self.snake.direction == DIRECTION_UP:
            if y_head == y and x_head == x - 1:
                return True
            else:
                return False
        elif self.snake.direction == DIRECTION_DOWN:
            if y_head == y and x_head == x + 1:
                return True
            else:
                return False
        elif self.snake.direction == DIRECTION_RIGHT:
            if x_head == x and y_head == y - 1:
                return True
            else:
                return False
        elif self.snake.direction == DIRECTION_LEFT:
            if x_head == x and y_head == y + 1:
                return True
            else:
                return False


    def part_left(self, x, y):
        x_head = self.snake.segments[0][0]
        y_head = self.snake.segments[0][1]
        if self.snake.direction == DIRECTION_UP:
            if y_head == y and x_head == x + 1:
                return True
            else:
                return False
        elif self.snake.direction == DIRECTION_DOWN:
            if y_head == y and x_head == x - 1:
                return True
            else:
                return False
        elif self.snake.direction == DIRECTION_RIGHT:
            if x_head == x and y_head == y + 1:
                return True
            else:
                return False
        elif self.snake.direction == DIRECTION_LEFT:
            if x_head == x and y_head == y - 1:
                return True
            else:
                return False


    def part_above(self, x, y):
        x_head = self.snake.segments[0][0]
        y_head = self.snake.segments[0][1]
        if self.snake.direction == DIRECTION_UP:
            if y_head == y + 1 and x_head == x:
                return True
            else:
                return False
        elif self.snake.direction == DIRECTION_DOWN:
            if y_head == y - 1 and x_head == x:
                return True
            else:
                return False
        elif self.snake.direction == DIRECTION_RIGHT:
            if x_head == x - 1 and y_head == y:
                return True
            else:
                return False
        elif self.snake.direction == DIRECTION_LEFT:
            if x_head == x + 1 and y_head == y:
                return True
            else:
                return False


    def part_above_right(self, x, y):
        x_head = self.snake.segments[0][0]
        y_head = self.snake.segments[0][1]
        if self.snake.direction == DIRECTION_UP:
            if y_head == y + 1 and x_head == x - 1:
                return True
            else:
                return False
        elif self.snake.direction == DIRECTION_DOWN:
            if y_head == y - 1 and x_head == x + 1:
                return True
            else:
                return False
        elif self.snake.direction == DIRECTION_RIGHT:
            if x_head == x - 1 and y_head == y - 1:
                return True
            else:
                return False
        elif self.snake.direction == DIRECTION_LEFT:
            if x_head == x + 1 and y_head == y + 1:
                return True
            else:
                return False


    def part_above_left(self, x, y):
        x_head = self.snake.segments[0][0]
        y_head = self.snake.segments[0][1]
        if self.snake.direction == DIRECTION_UP:
            if y_head == y + 1 and x_head == x + 1:
                return True
            else:
                return False
        elif self.snake.direction == DIRECTION_DOWN:
            if y_head == y - 1 and x_head == x - 1:
                return True
            else:
                return False
        elif self.snake.direction == DIRECTION_RIGHT:
            if x_head == x - 1 and y_head == y + 1:
                return True
            else:
                return False
        elif self.snake.direction == DIRECTION_LEFT:
            if x_head == x + 1 and y_head == y - 1:
                return True
            else:
                return False

    def wall_right(self):
        if self.snake.direction == DIRECTION_RIGHT:
            if self.snake.segments[0][1] == self.size - 1:
                return True
            return False
        if self.snake.direction == DIRECTION_LEFT:
            if self.snake.segments[0][1] == 0:
                return True
            return False
        if self.snake.direction == DIRECTION_UP:
            if self.snake.segments[0][0] == self.size - 1:
                return True
            return False
        if self.snake.direction == DIRECTION_DOWN:
            if self.snake.segments[0][0] == 0:
                return True
            return False

    def wall_left(self):
        if self.snake.direction == DIRECTION_LEFT:
            if self.snake.segments[0][1] == self.size - 1:
                return True
            return False
        if self.snake.direction == DIRECTION_RIGHT:
            if self.snake.segments[0][1] == 0:
                return True
            return False
        if self.snake.direction == DIRECTION_DOWN:
            if self.snake.segments[0][0] == self.size - 1:
                return True
            return False
        if self.snake.direction == DIRECTION_UP:
            if self.snake.segments[0][0] == 0:
                return True
            return False


    def rightLeft(self):
        block_meeting_right = False
        block_meeting_left = False
        for part in self.snake.segments:
            # if block right
            if self.part_right(part[0], part[1]) or self.wall_right() and not block_meeting_right:
                block_meeting_right = True
            # if block left
            if self.part_left(part[0], part[1]) or self.wall_left() and not block_meeting_left:
                block_meeting_left = True

        return block_meeting_right and block_meeting_left



    def only3face(self):
        block_meeting_cond = False
        for part in self.snake.segments:
            # if block right
            if self.part_right(part[0], part[1]):
                return False
            # if block left
            if self.part_left(part[0], part[1]):
                return False

            if not block_meeting_cond:
                if self.part_above(part[0], part[1]):
                    block_meeting_cond = True
                if self.part_above_right(part[0], part[1]):
                    block_meeting_cond = True
                if self.part_above_left(part[0], part[1]):
                    block_meeting_cond = True
        # todo : need to replace things here to redo the training correctly
        # wall can count for block_meeting_cond too
        # right wall, but we cannot go along it :
        if self.snake.segments[0][0] == self.size - 1 \
                and self.snake.direction != DIRECTION_UP \
                and self.snake.direction != DIRECTION_DOWN:
            block_meeting_cond = True
        # left wall, but we cannot go along it :
        if self.snake.segments[0][0] == 0\
                and self.snake.direction != DIRECTION_UP\
                and self.snake.direction != DIRECTION_DOWN:
            block_meeting_cond = True
        # above wall, but we cannot go along it :
        if self.snake.segments[0][1] == 0\
                and self.snake.direction != DIRECTION_LEFT\
                and self.snake.direction != DIRECTION_RIGHT:
            block_meeting_cond = True
        # under wall, but we cannot go along it :
        if self.snake.segments[0][1] == self.size - 1\
                and self.snake.direction != DIRECTION_LEFT\
                and self.snake.direction != DIRECTION_RIGHT:
            block_meeting_cond = True

        return block_meeting_cond


    def upRnoR(self):
        block_meeting_cond = False
        for part in self.snake.segments:
            # if block right
            if self.part_right(part[0], part[1]):
                return False
            # if block above
            if self.part_above(part[0], part[1]):
                return False
            # if block meeting cond is already true, no need to re-calculate every thing
            if not block_meeting_cond:
                if self.part_above_right(part[0], part[1]):
                    block_meeting_cond = True

        return block_meeting_cond

    def upLnoL(self):
        block_meeting_cond = False
        for part in self.snake.segments:
            # if block left
            if self.part_left(part[0], part[1]):
                return False
            if self.part_above(part[0], part[1]):
                return False
            # if block meeting cond is already true, no need to re-calculate every thing
            if not block_meeting_cond:
                if self.part_above_left(part[0], part[1]):
                    block_meeting_cond = True
        return block_meeting_cond

    def breadth_countL(self, x_dep, y_dep):
        global free_block_left
        if not free_block_left[y_dep+1][x_dep+1]:
            return 0

        global block_encountered_left
        block_encountered_left[y_dep+1][x_dep+1] = 1
        free_block_left[y_dep+1][x_dep+1] = False
        count_above = self.breadth_countL(x_dep, y_dep-1)
        count_down = self.breadth_countL(x_dep, y_dep+1)
        count_right = self.breadth_countL(x_dep+1, y_dep)
        count_left = self.breadth_countL(x_dep-1, y_dep)

        return 1 + count_above + count_down + count_left + count_right

    def breadth_countA(self, x_dep, y_dep):
        global free_block_above
        if not free_block_above[y_dep+1][x_dep+1]:
            return 0

        global block_encountered_above
        block_encountered_above[y_dep+1][x_dep+1] = 1
        free_block_above[y_dep+1][x_dep+1] = False
        count_above = self.breadth_countA(x_dep, y_dep-1)
        count_down = self.breadth_countA(x_dep, y_dep+1)
        count_right = self.breadth_countA(x_dep+1, y_dep)
        count_left = self.breadth_countA(x_dep-1, y_dep)

        return 1 + count_above + count_down + count_left + count_right

    def breadth_countR(self, x_dep, y_dep):
        global free_block_right
        if not free_block_right[y_dep+1][x_dep+1]:
            return 0

        global block_encountered_right
        block_encountered_right[y_dep+1][x_dep+1] = 1
        free_block_right[y_dep+1][x_dep+1] = False
        count_above = self.breadth_countR(x_dep, y_dep-1)
        count_down = self.breadth_countR(x_dep, y_dep+1)
        count_right = self.breadth_countR(x_dep+1, y_dep)
        count_left = self.breadth_countR(x_dep-1, y_dep)

        return 1 + count_above + count_down + count_left + count_right

    def make_0_from_0(self, zeros):  # returns all the zeros that are not already colored
        new_zeros = []
        global free_block_above
        global free_block_left
        global free_block_right
        matrices = [free_block_above, free_block_left, free_block_right]

        for matrix in matrices:
            for coord in zeros:
                # above
                if matrix[coord[1]][coord[0]+1] == 1:
                    new_zeros.append((coord[0], coord[1]-1))
                    matrix[coord[1]][coord[0] + 1] = 0

                # under
                if matrix[coord[1]+2][coord[0]+1] == 1:
                    new_zeros.append((coord[0], coord[1]+1))
                    matrix[coord[1]+2][coord[0] + 1] = 0

                # right
                if matrix[coord[1] + 1][coord[0] + 2] == 1:
                    new_zeros.append((coord[0]+1, coord[1]))
                    matrix[coord[1] + 1][coord[0] + 2] = 0

                # left
                if matrix[coord[1] + 1][coord[0]] == 1:
                    new_zeros.append((coord[0]-1, coord[1]))
                    matrix[coord[1] + 1][coord[0]] = 0

        return new_zeros

    def sonar(self):
        global free_block_above
        global free_block_left
        global free_block_right

        n_tones = 9 # must always be an odd number
        cubes_diff_light = []
        for i in range(n_tones):
            cubes_diff_light.append([])
        cubes_diff_light = np.array(cubes_diff_light)

        x_head = self.snake.segments[0][0]
        y_head = self.snake.segments[0][1]
        zeros = [(x_head, y_head)]
        new_zeros = self.make_0_from_0(zeros)
        cubes_to_light = True
        while cubes_to_light:
            new_cubes_diff = [new_zeros]
            for i in range(n_tones-1):
                new_cubes_diff.append(cubes_diff_light[i])
            cubes_diff_light = np.array(new_cubes_diff, dtype=object)

            self.draw_sonar(self.n_eval, cubes_diff_light)
            time.sleep(0.03)
            new_zeros = self.make_0_from_0(new_zeros)
            if len(cubes_diff_light[0]) == 0 \
                    and len(cubes_diff_light[1]) == 0 \
                    and len(cubes_diff_light[2]) == 0 \
                    and len(cubes_diff_light[3]) == 0 \
                    and len(cubes_diff_light[4]) == 0:
                cubes_to_light = False



    def choose_direction(self):
        # action[0] = 1 : next left
        # action[1] = 1 : next middle
        # action[2] = 1 : next right

        # determining the blocs left, above and right of the head
        x_right = 0
        y_right = 0
        x_left = 0
        y_left = 0
        x_above = 0
        y_above = 0
        x_head = self.snake.segments[0][0]
        y_head = self.snake.segments[0][1]
        if self.snake.direction == DIRECTION_UP:
            x_right = x_head + 1
            y_right = y_head
            x_left = x_head - 1
            y_left = y_head
            x_above = x_head
            y_above = y_head - 1
        elif self.snake.direction == DIRECTION_DOWN:
            x_right = x_head - 1
            y_right = y_head
            x_left = x_head + 1
            y_left = y_head
            x_above = x_head
            y_above = y_head + 1
        elif self.snake.direction == DIRECTION_RIGHT:
            x_right = x_head
            y_right = y_head + 1
            x_left = x_head
            y_left = y_head - 1
            x_above = x_head + 1
            y_above = y_head
        elif self.snake.direction == DIRECTION_LEFT:
            x_right = x_head
            y_right = y_head - 1
            x_left = x_head
            y_left = y_head + 1
            x_above = x_head - 1
            y_above = y_head

        global free_block_right
        free_block_right = np.ones((self.size+2, self.size+2), dtype=bool)
        # setting the walse
        for i in range(len(free_block_right[0])):
            free_block_right[0][i] = False
            free_block_right[-1][i] = False
            free_block_right[i][0] = False
            free_block_right[i][-1] = False
        # setting the snake blocks
        for part in self.snake.segments:
            free_block_right[part[1]+1][part[0]+1] = False
        global free_block_above
        free_block_above = np.copy(free_block_right)
        global free_block_left
        free_block_left = np.copy(free_block_right)

        global block_encountered_right
        global block_encountered_above
        global block_encountered_left
        block_encountered_right = np.ones((self.size+2, self.size+2)) * (-1)
        block_encountered_left = np.ones((self.size+2, self.size+2)) * (-1)
        block_encountered_above = np.ones((self.size+2, self.size+2)) * (-1)

        # head at 0 => useless
        block_encountered_right[y_head+1][x_head+1] = 0
        block_encountered_left[y_head+1][x_head+1] = 0
        block_encountered_above[y_head+1][x_head+1] = 0

        # adding all the blocks in order of visit in lists

        count_left = self.breadth_countL(x_left, y_left)
        count_above = self.breadth_countA(x_above, y_above)
        count_right = self.breadth_countR(x_right, y_right)

        # visualising the breadth first count
        if self.visualising_sonar:
            self.sonar()

        return [count_left, count_above, count_right]

    def infinite_wall(self):
        x_head = self.snake.segments[0][0]
        y_head = self.snake.segments[0][1]

        # right wall
        if x_head == self.size - 1 and self.snake.direction == DIRECTION_UP:
            return True
        # left wall
        if x_head == 0 and self.snake.direction == DIRECTION_DOWN:
            return True
        # up wall
        if y_head == 0 and self.snake.direction == DIRECTION_LEFT:
            return True
        # under wall
        if y_head == self.size - 1 and self.snake.direction == DIRECTION_RIGHT:
            return True
        return False

    def process_infinite_wall(self):
        global de_infinite
        x_head = self.snake.segments[0][0]
        y_head = self.snake.segments[0][1]

        # still hasn't reached the end of the wall and no obstacle
        # left wall
        if x_head == 0 and y_head != self.size - 1 \
                and self.snake.direction == DIRECTION_DOWN\
                and y_head != 1:
            #print("still hasn't reached the end of the wall : left")
            return [0, 1, 0]
        # right wall
        if x_head == self.size - 1 and y_head != 0 \
                and self.snake.direction == DIRECTION_UP\
                and y_head != self.size - 2:
            #print("still hasn't reached the end of the wall : right")
            return [0, 1, 0]
        # above wall
        if y_head == 0 and x_head != 0 \
                and self.snake.direction == DIRECTION_LEFT\
                and x_head != self.size - 2:
            #print("still hasn't reached the end of the wall : above")
            return [0, 1, 0]
        # under wall
        if y_head == self.size - 1 and x_head != self.size - 1 \
                and self.snake.direction == DIRECTION_RIGHT\
                and x_head != 1:
            #print("still hasn't reached the end of the wall : under")
            return [0, 1, 0]

        # has reached the end of the wall

        if (x_head == 0 or x_head == self.size-1) and (y_head == 0 or y_head == self.size - 1):
            # print("reached the end of the first wall")
            return [1, 0, 0]

        # need to turn a second time
        # right wall
        if self.snake.direction == DIRECTION_LEFT and x_head == self.size - 2 and y_head == 0:
            # print("turned a second time : right wall")
            return [1, 0, 0]
        # left wall
        if self.snake.direction == DIRECTION_RIGHT and x_head == 1 and y_head == self.size - 1:
            # print("turned a second time : left wall")
            return [1, 0, 0]
        # above wall
        if self.snake.direction == DIRECTION_DOWN and x_head == 0 and y_head == 1:
            # print("turned a second time : above wall")
            return [1, 0, 0]
        # down wall
        if self.snake.direction == DIRECTION_UP and x_head == self.size - 1 and y_head == self.size - 2:
            # print("turned a second time : down wall")
            return [1, 0, 0]


        # if no block just in front or food exactly right
        # block
        for part in self.snake.segments:
            if self.part_above(part[0], part[1]):
                de_infinite = False
                #print("part just in front")
                return [0, 0, 1]
        # food
        food_x = food[0]
        food_y = food[1]
        # right/left wall
        if (x_head == 1 or x_head == self.size - 2) and (y_head == food_y):
            #print("no more in infinite wall")
            de_infinite = False
            return [0, 0, 1]
        # above/under wall
        if (y_head == 1 or y_head == self.size - 2) and (x_head == food_x):
            #print("no more in infinite wall")
            de_infinite = False
            return [0, 0, 1]

        # still going counterflow of the wall
        #print("still going counterflow")
        return [0, 1, 0]


    @property
    def play(self):
        global time_passes
        global tik
        """Play game until the QUIT event is received."""
        eval = 1
        tik = 1 / self.snake.speed
        just_choosed_direction = True
        while True:
            if just_choosed_direction:
                dt = self.clock.tick(FPS) / 1000.0
                dt = 0.016
                just_choosed_direction = False
            else:
                dt = self.clock.tick(FPS) / 1000.0  # convert to seconds => number of seconds since last clocktick

            for e in pygame.event.get():
                if e.type == QUIT:
                    return
                elif e.type == KEYUP:
                    self.input(e)

            if self.moves == 0:
                self.snake.brain.score += self.score

                if eval == self.n_eval:
                    eval = 1
                    self.brains[self.batch] = self.snake.brain
                    self.batch += 1
                    self.reset(self.parents, self.scores_p, self.proportion, self.amplitude,
                               self.batch, self.speed, self.loaded, self.structure, same=False)
                    if self.batch == self.n_batch:
                        return self.brains
                else:
                    eval += 1
                    self.reset(self.parents, self.scores_p, self.proportion, self.amplitude,
                               self.batch, self.speed, self.loaded, self.structure, same=True)
                self.moves = self.save_moves

            if self.playing:
                if time_passes:
                    self.update(dt)
                    tik -= dt

                if tik < 0:
                    tik += 1 / self.snake.speed
                    self.moves -= 1
                    # initiating
                    cos_food = 0
                    direction = 0
                    walls = 0
                    # determining the cosinus of the angle between the food and the head
                    hyp = ((self.snake.segments[0][0] - food[0]) ** 2 +
                           (self.snake.segments[0][1] - food[1]) ** 2) ** (1 / 2)
                    if self.snake.direction == DIRECTION_UP:
                        if food[1] - self.snake.segments[0][1] > 0:
                            direction = -1
                        elif food[1] - self.snake.segments[0][1] == 0:
                            direction = 0
                        else:
                            direction = 1

                        adj = abs(self.snake.segments[0][0] - food[0])
                        if food[0] - self.snake.segments[0][0] > 0:
                            cos_food = adj / hyp
                        else:
                            cos_food = -adj / hyp

                    if self.snake.direction == DIRECTION_DOWN:
                        if food[1] - self.snake.segments[0][1] > 0:
                            direction = 1
                        elif food[1] - self.snake.segments[0][1] == 0:
                            direction = 0
                        else:
                            direction = -1

                        adj = abs(self.snake.segments[0][0] - food[0])
                        if food[0] - self.snake.segments[0][0] > 0:
                            cos_food = -adj / hyp
                        else:
                            cos_food = adj / hyp

                    if self.snake.direction == DIRECTION_RIGHT:
                        if food[0] - self.snake.segments[0][0] > 0:
                            direction = 1
                        elif food[0] - self.snake.segments[0][0] == 0:
                            direction = 0
                        else:
                            direction = -1

                        adj = abs(self.snake.segments[0][1] - food[1])
                        if food[1] - self.snake.segments[0][1] > 0:
                            cos_food = adj / hyp
                        else:
                            cos_food = -adj / hyp

                    if self.snake.direction == DIRECTION_LEFT:
                        if food[0] - self.snake.segments[0][0] > 0:
                            direction = -1
                        elif food[0] - self.snake.segments[0][0] == 0:
                            direction = 0
                        else:
                            direction = 1

                        adj = abs(self.snake.segments[0][1] - food[1])
                        if food[1] - self.snake.segments[0][1] > 0:
                            cos_food = -adj / hyp
                        else:
                            cos_food = adj / hyp





                    # determining if there are body parts next to the head
                    batch_count = 0  # the 3 first blocks doesn't matter
                    ur = 0
                    um = 0
                    ul = 0
                    ml = 0
                    al = 0
                    am = 0
                    ar = 0
                    mr = 0
                    for parts in self.snake.segments:
                        if batch_count < 3:
                            batch_count += 1
                        else:
                            x_head = self.snake.segments[0][0]
                            y_head = self.snake.segments[0][1]

                            if x_head - parts[0] == -1 and y_head - parts[1] == -1:
                                ur = 1  # under_right  x_y 1

                            if x_head - parts[0] == 0 and y_head - parts[1] == -1:
                                um = 1  # under_middle 2

                            if x_head - parts[0] == 1 and y_head - parts[1] == -1:
                                ul = 1  # under_left 3

                            if x_head - parts[0] == 1 and y_head - parts[1] == 0:
                                ml = 1  # middle_left 4

                            if x_head - parts[0] == 1 and y_head - parts[1] == 1:
                                al = 1  # above_left 5

                            if x_head - parts[0] == 0 and y_head - parts[1] == 1:
                                am = 1  # above_middle 6

                            if x_head - parts[0] == -1 and y_head - parts[1] == 1:
                                ar = 1  # above_right 7

                            if x_head - parts[0] == -1 and y_head - parts[1] == 0:
                                mr = 1  # middle_right 8

                            walls = np.array([ur, um, ul, ml, al, am, ar, mr])
                            # implem wall
                            # right wall :
                            if self.snake.segments[0][0] == self.size - 1:
                                walls[0] = 1
                                walls[6] = 1
                                walls[7] = 1
                            # left wall :
                            if self.snake.segments[0][0] == 0:
                                walls[2] = 1
                                walls[3] = 1
                                walls[4] = 1
                            # above wall :
                            if self.snake.segments[0][1] == 0:
                                walls[4] = 1
                                walls[5] = 1
                                walls[6] = 1
                            # under wall :
                            if self.snake.segments[0][1] == self.size - 1:
                                walls[0] = 1
                                walls[1] = 1
                                walls[2] = 1


                            if self.snake.direction == DIRECTION_RIGHT:
                                walls = np.concatenate((walls[2:], walls[:2]))

                            elif self.snake.direction == DIRECTION_DOWN:
                                walls = np.concatenate((walls[4:], walls[:4]))

                            elif self.snake.direction == DIRECTION_LEFT:
                                walls = np.concatenate((walls[6:], walls[:-2]))
                            # on prend pas les 3 premiers elements du walls
                            # en clair on prend que les 5 éléments devant le snake
                            walls = walls[3:]





                    if self.use_sonar:
                        global de_infinite
                        if self.infinite_wall() or de_infinite:
                            de_infinite = True
                            actions = self.process_infinite_wall()
                        else:
                            if self.only3face():
                                actions = self.choose_direction()
                                just_choosed_direction = True

                                # let the brain choose if equality
                                # count left & above same and bigger than right
                                if actions[0] == actions[1] and actions[0] > actions[2]:
                                    obs = np.concatenate(([cos_food], [direction], walls))
                                    actions = self.snake.brain.think(obs)

                                # count left & right same and bigger than count above
                                if actions[0] == actions[2] and actions[0] > actions[1]:
                                    # todo : changer cette cond si on réentraine le snake
                                    actions = [0, 0, 1]

                                # count right & above same and bigger than left
                                if actions[1] == actions[2] and actions[1] > actions[0]:
                                    obs = np.concatenate(([cos_food], [direction], walls))
                                    actions = self.snake.brain.think(obs)
                            elif self.upRnoR():  # no block up too
                                # print("on est dans la cond un en haut à droite")
                                actions = self.choose_direction()
                                just_choosed_direction = True

                                # let the brain choose if equality
                                # count left & above same and bigger than right
                                if actions[0] == actions[1] and actions[0] > actions[2]:
                                    # print("the brain is choosing")
                                    obs = np.concatenate(([cos_food], [direction], walls))
                                    actions = self.snake.brain.think(obs)

                                # count left & right same and bigger than count above
                                if actions[0] == actions[2] and actions[0] > actions[1]:
                                    # print("the brain is choosing")
                                    obs = np.concatenate(([cos_food], [direction], walls))
                                    actions = self.snake.brain.think(obs)

                                # count right & above same and bigger than left
                                if actions[1] == actions[2] and actions[1] > actions[0]:
                                    # print("the brain is choosing")
                                    obs = np.concatenate(([cos_food], [direction], walls))
                                    actions = self.snake.brain.think(obs)
                            elif self.upLnoL():  # no block up too
                                # print("on est dans la cond un en haut à gauche")
                                actions = self.choose_direction()
                                just_choosed_direction = True

                                # let the brain choose if equality
                                # count left & above same and bigger than right
                                if actions[0] == actions[1] and actions[0] > actions[2]:
                                    # print("the brain is choosing")
                                    obs = np.concatenate(([cos_food], [direction], walls))
                                    actions = self.snake.brain.think(obs)

                                # count left & right same and bigger than count above
                                if actions[0] == actions[2] and actions[0] > actions[1]:
                                    # print("the brain is choosing")
                                    obs = np.concatenate(([cos_food], [direction], walls))
                                    actions = self.snake.brain.think(obs)

                                # count right & above same and bigger than left
                                if actions[1] == actions[2] and actions[1] > actions[0]:
                                    # print("the brain is choosing")
                                    obs = np.concatenate(([cos_food], [direction], walls))
                                    actions = self.snake.brain.think(obs)
                            else:  # il n'y a pas de danger de se faire enrouler
                                if self.rightLeft():
                                    actions = [0, 1, 0]
                                else:
                                    obs = np.concatenate(([cos_food], [direction], walls))
                                    actions = self.snake.brain.think(obs)

                        actions = choice(actions)
                        self.brain_action(actions)

                    else:
                        obs = np.concatenate(([cos_food], [direction], walls))
                        actions = self.snake.brain.think(obs)
                        actions = choice(actions)
                        self.brain_action(actions)

                if self.see:
                    self.draw(eval=eval)
            else:
                #time.sleep(5)
                print(self.score)
                self.snake.brain.score += self.score
                self.brains[self.batch] = self.snake.brain

                if eval == self.n_eval:
                    eval = 1
                    self.brains[self.batch] = self.snake.brain
                    self.batch += 1
                    self.reset(self.parents, self.scores_p, self.proportion, self.amplitude,
                               self.batch, self.speed, self.loaded, self.structure, same=False)
                    if self.batch == self.n_batch:
                        return self.brains
                else:
                    eval += 1
                    self.reset(self.parents, self.scores_p, self.proportion, self.amplitude,
                               self.batch, self.speed, self.loaded, self.structure, same=True)

                self.moves = self.save_moves

            if self.see:
                pygame.display.flip()
