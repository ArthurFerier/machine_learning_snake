# on fait les modifs ici pour le machine learning
from collections import deque
import pygame
from random import randrange
from pygame.locals import *
from réseaux.Multi_layer_NN import *


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


FPS = 60                        # Game frames per second
SEGMENT_SCORE = 1               # Score per segment

SNAKE_SPEED_INCREMENT = 0.25    # Snake speeds up this much each time it grows
SNAKE_START_LENGTH = 5          # Initial snake length in segments

WORLD_SIZE = Vector((20, 20))   # World size, in blocks
BLOCK_SIZE = 24                 # Block size, in pixels

BACKGROUND_COLOR = 0, 0, 0
SNAKE_COLOR = 255, 255, 255
FOOD_COLOR = 255, 0, 0
DEATH_COLOR = 255, 0, 0
TEXT_COLOR = 255, 255, 255

DIRECTION_UP    = Vector((0, -1))
DIRECTION_DOWN  = Vector((0,  1))
DIRECTION_LEFT  = Vector((-1,  0))
DIRECTION_RIGHT = Vector((1,  0))
DIRECTION_DR    = DIRECTION_DOWN + DIRECTION_RIGHT

# Map from PyGame key event to the corresponding direction.
KEY_DIRECTION = {
    K_q: DIRECTION_UP,    K_UP:    DIRECTION_UP,
    K_s: DIRECTION_DOWN,  K_DOWN:  DIRECTION_DOWN,
    K_a: DIRECTION_LEFT,  K_LEFT:  DIRECTION_LEFT,
    K_d: DIRECTION_RIGHT, K_RIGHT: DIRECTION_RIGHT,
}


class Snake(object):
    def __init__(self, start, start_length, pot_parents, scores_p, proportion, amplitude, batch, speed, loaded, struct, amplitude_init):
        self.speed = speed                # Speed in squares per second.
        self.timer = 1.0 / self.speed     # Time remaining to next movement.
        self.growth_pending = 0           # Number of segments still to grow.
        self.direction = DIRECTION_UP     # Current movement direction.
        self.segments = deque([start - self.direction * i for i in range(start_length)])
        # juste une liste de tuple contenant les coordonnées des blocks du snake [(xhead, yhead), (xsec, ysec),...]
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
                self.brain = MLNeuralNetwork(structure, amplitude_init)
            else:
                parents = pooling(pot_parents, scores_p).tolist()
                self.brain = MLNeuralNetwork(parents)
                self.brain.mutate(proportion, amplitude)
                """
                if batch == 0:
                    self.brain = pot_parents[0]
                else:
                    parents = pooling(pot_parents, scores_p).tolist()
                    self.brain = MLNeuralNetwork(parents)
                    self.brain.mutate(proportion, amplitude)"""

    def __iter__(self):
        return iter(self.segments)

    def __len__(self):
        return len(self.segments)

    def head(self):
        """Return the position of the snake's head."""
        return self.segments[0]

    def update(self, dt, direction):                      # dans cette fonction que je dois faire le brain
        """Update the snake by dt seconds and possibly set direction."""
        self.timer -= dt
        if self.timer > 0:
            # Nothing to do yet.
            return

        #if self.direction != -direction:
            #self.direction = direction

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

    def self_intersecting(self):
        """Is the snake currently self-intersecting?"""
        it = iter(self)
        head = next(it)
        return head in it


class SnakeGame(object):
    def __init__(self, parents, scores_p, structure,
                 proportion, amplitude, moves, add_moves,
                 generation, screen, speed, size, loaded, n_batch, amplitude_init=1):
        self.amplitude_init = amplitude_init
        pygame.display.set_caption('PyGame Snake')
        self.block_size = BLOCK_SIZE
        self.see = screen
        if self.see:
            self.window = pygame.display.set_mode(Vector((size, size)) * self.block_size) #turn off if doesn't want to see the screen
        self.screen = pygame.display.get_surface()
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font('freesansbold.ttf', 20)
        self.world = Rect((0, 0), Vector((size, size)))
        self.reset(parents, scores_p, proportion, amplitude, 0, speed, loaded, structure)
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


    def reset(self, parents, scores_p, proportion, amplitude, batch, speed, loaded, structure):
        """Start a new game."""
        self.playing = True
        self.next_direction = DIRECTION_UP
        self.score = 0
        self.snake = Snake(self.world.center, SNAKE_START_LENGTH,
                           parents, scores_p, proportion,
                           amplitude, batch, speed, loaded, structure, self.amplitude_init)
        # à modifier en boucle for
        self.food = set()
        self.add_food()

    def add_food(self):
        """Ensure that there is at least one piece of food.
        (And, with small probability, more than one.)
        """
        while not (self.food and randrange(4)):
            global food
            food = Vector(map(randrange, self.world.bottomright))
            if food not in self.food and food not in self.snake and food[0] != self.world.center[0]:
                self.food.add(food)
                break  # comme ça il en crée pas plusieurs ce con

    def input(self, e):
        """Process keyboard event e."""
        if e.key in KEY_DIRECTION:
            self.next_direction = KEY_DIRECTION[e.key]


    def brain_action(self, actions):
        #print(actions)

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
            self.score += 1 #len(self.snake) * SEGMENT_SCORE
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
        self.screen.blit(self.font.render(text, 1, TEXT_COLOR), p)

    def draw(self):
        """Draw game (while playing)."""
        self.screen.fill(BACKGROUND_COLOR)
        for p in self.snake:
            pygame.draw.rect(self.screen, SNAKE_COLOR, self.block(p))
        for f in self.food:
            pygame.draw.rect(self.screen, FOOD_COLOR, self.block(f))
        self.draw_text("Score: {}".format(self.score), (20, 20))
        self.draw_text("Moves left: {}".format(self.moves), (20, 40))
        self.draw_text("generation: {}".format(self.generation), (20, 60))
        self.draw_text("batch: {}".format(self.batch), (20, 80))

    def draw_death(self):
        """Draw game (after game over)."""
        self.screen.fill(DEATH_COLOR)
        self.draw_text("Game over! Press Space to start a new game", (20, 150))
        self.draw_text("Your score is: {}".format(self.score), (140, 180))

    @property
    def play(self):
        """Play game until the QUIT event is received."""
        tik = 1/self.snake.speed
        while True:
            dt = self.clock.tick(FPS) / 1000.0  # convert to seconds

            for e in pygame.event.get():
                if e.type == QUIT:
                    return
                elif e.type == KEYUP:
                    self.input(e)

            if self.moves == 0:
                self.snake.brain.score = self.score
                self.brains[self.batch] = self.snake.brain
                self.batch += 1
                if self.batch == self.n_batch:
                    return self.brains
                self.reset(self.parents, self.scores_p, self.proportion, self.amplitude,
                           self.batch, self.speed, self.loaded, self.structure)
                self.moves = self.save_moves


            if self.playing:
                self.update(dt)
                tik -= dt

                if tik < 0:
                    tik += 1/self.snake.speed
                    self.moves -= 1

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
                            cos_food = adj/hyp
                        else:
                            cos_food = -adj/hyp

                    elif self.snake.direction == DIRECTION_DOWN:
                        if food[1] - self.snake.segments[0][1] > 0:
                            direction = 1
                        elif food[1] - self.snake.segments[0][1] == 0:
                            direction = 0
                        else:
                            direction = -1

                        adj = abs(self.snake.segments[0][0] - food[0])
                        if food[0] - self.snake.segments[0][0] > 0:
                            cos_food = -adj/hyp
                        else:
                            cos_food = adj/hyp

                    if self.snake.direction == DIRECTION_RIGHT:
                        if food[0] - self.snake.segments[0][0] > 0:
                            direction = 1
                        elif food[0] - self.snake.segments[0][0] == 0:
                            direction = 0
                        else:
                            direction = -1

                        adj = abs(self.snake.segments[0][1] - food[1])
                        if food[1] - self.snake.segments[0][1] > 0:
                            cos_food = adj/hyp
                        else:
                            cos_food = -adj/hyp

                    elif self.snake.direction == DIRECTION_LEFT:
                        if food[0] - self.snake.segments[0][0] > 0:
                            direction = -1
                        elif food[0] - self.snake.segments[0][0] == 0:
                            direction = 0
                        else:
                            direction = 1

                        adj = abs(self.snake.segments[0][1] - food[1])
                        if food[1] - self.snake.segments[0][1] > 0:
                            cos_food = -adj/hyp
                        else:
                            cos_food = adj/hyp
                    #print(cos_food)
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
                            if self.snake.direction == DIRECTION_RIGHT:
                                walls = np.concatenate((walls[2:], walls[:2]))

                            elif self.snake.direction == DIRECTION_DOWN:
                                walls = np.concatenate((walls[4:], walls[:4]))

                            elif self.snake.direction == DIRECTION_LEFT:
                                walls = np.concatenate((walls[6:], walls[:-2]))
                            walls = walls[3:]

                    obs = np.concatenate(([cos_food], [direction], walls))
                    actions = self.snake.brain.think(obs)
                    actions = choice(actions)

                    self.brain_action(actions)

                if self.see:
                    self.draw()
            else:
                self.snake.brain.score = self.score
                self.brains[self.batch] = self.snake.brain
                self.batch += 1
                if self.batch == self.n_batch:
                    return self.brains  # TypeError: 'numpy.ndarray' object is not callable
                self.reset(self.parents, self.scores_p, self.proportion, self.amplitude,
                           self.batch, self.speed, self.loaded, self.structure)
                self.moves = self.save_moves

            if self.see:
                pygame.display.flip()
