import time

import gym
from gym import spaces
from save_snake import *

N_DISCRETE_ACTIONS = 3  # can stay in the direction (0) or left(1)/right(2)

class SnekEnv(gym.Env):
    """Custom Environment that follows gym interface"""

    def determine_cos(self):
        cos_food = None


        hyp = ((self.snakeGame.snake.segments[0][0] - self.snakeGame.food[0][0]) ** 2 +
               (self.snakeGame.snake.segments[0][1] - self.snakeGame.food[0][1]) ** 2) ** (1 / 2)

        if self.snakeGame.snake.direction == DIRECTION_UP:
            adj = abs(self.snakeGame.snake.segments[0][0] - self.snakeGame.food[0][0])
            if self.snakeGame.food[0][0] - self.snakeGame.snake.segments[0][0] > 0:
                cos_food = adj / hyp
            else:
                cos_food = -adj / hyp

        if self.snakeGame.snake.direction == DIRECTION_DOWN:
            adj = abs(self.snakeGame.snake.segments[0][0] - self.snakeGame.food[0][0])
            if self.snakeGame.food[0][0] - self.snakeGame.snake.segments[0][0] > 0:
                cos_food = -adj / hyp
            else:
                cos_food = adj / hyp

        if self.snakeGame.snake.direction == DIRECTION_RIGHT:
            adj = abs(self.snakeGame.snake.segments[0][1] - self.snakeGame.food[0][1])
            if self.snakeGame.food[0][1] - self.snakeGame.snake.segments[0][1] > 0:
                cos_food = adj / hyp
            else:
                cos_food = -adj / hyp

        if self.snakeGame.snake.direction == DIRECTION_LEFT:
            adj = abs(self.snakeGame.snake.segments[0][1] - self.snakeGame.food[0][1])
            if self.snakeGame.food[0][1] - self.snakeGame.snake.segments[0][1] > 0:
                cos_food = -adj / hyp
            else:
                cos_food = adj / hyp

        return cos_food


    def determine_dists(self):
        walls = None

        snake = self.snakeGame.snake

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
        for parts in snake.segments:
            if batch_count < 3:
                batch_count += 1
            else:
                x_head = snake.segments[0][0]
                y_head = snake.segments[0][1]

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
                if snake.segments[0][0] == self.snakeGame.size - 1:
                    walls[0] = 1
                    walls[6] = 1
                    walls[7] = 1
                # left wall :
                if snake.segments[0][0] == 0:
                    walls[2] = 1
                    walls[3] = 1
                    walls[4] = 1
                # above wall :
                if snake.segments[0][1] == 0:
                    walls[4] = 1
                    walls[5] = 1
                    walls[6] = 1
                # under wall :
                if snake.segments[0][1] == self.snakeGame.size - 1:
                    walls[0] = 1
                    walls[1] = 1
                    walls[2] = 1

                if snake.direction == DIRECTION_RIGHT:
                    walls = np.concatenate((walls[2:], walls[:2]))

                elif snake.direction == DIRECTION_DOWN:
                    walls = np.concatenate((walls[4:], walls[:4]))

                elif snake.direction == DIRECTION_LEFT:
                    walls = np.concatenate((walls[6:], walls[:-2]))
                # on prend pas les 3 premiers elements du walls
                # en clair on prend que les 5 éléments devant le snake
                walls = walls[3:]

        return walls


    def determine_space(self):

        if self.snakeGame.only3face():
            countLeft, countAbove, countRight = self.snakeGame.choose_direction()
            maxCount = max(countLeft, countAbove, countRight)

            spaceAbove = countAbove / maxCount
            spaceRight = countRight / maxCount
            spaceLeft = countLeft / maxCount

            return spaceLeft, spaceRight, spaceAbove
        else:
            return 1, 1, 1


    def makeObservation(self):
        if self.done:
            return [0, 0, 0, 0, 0, 0, 1, 1]

        cos = self.determine_cos()
        dLeft, dLeftFront, dFront, dRightFront, dRight = self.determine_dists()
        spaceLeft, spaceRight, spaceAbove = self.determine_space()

        return np.array([cos, dLeft, dLeftFront, dFront, dRightFront, dRight, spaceLeft, spaceRight, spaceAbove])


    def __init__(self):
        pygame.init()
        super(SnekEnv, self).__init__()
        self.snakeGame = SnakeGame()
        self.snakeGame.draw()
        pygame.display.flip()

        self.previous_length = None
        self.prev_reward = None
        self.done = None

        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)

        # Example for using image as input (channel-first; channel-last also works):
        self.observation_space = spaces.Box(low=-1, high=1,
                                            shape=(9,), dtype=float)



    def calculate_dist_to_food(self):
        return ((self.snakeGame.snake.segments[0][0] - self.snakeGame.food[0][0]) ** 2
                + (self.snakeGame.snake.segments[0][1] - self.snakeGame.food[0][1]) ** 2) ** (1 / 2)


    def food_eaten(self):
        if len(self.snakeGame.snake.segments) > self.previous_length:
            self.previous_length = len(self.snakeGame.snake.segments)
            return 1
        else:
            return 0


    def define_reward(self):

        # todo : define a reward for getting to the food quickly
        #        dynamically in function of the length of the snake
        #        the bigger the snake, the less penality for getting to the food slowly


        reward = 0

        coeff_dist = 5
        coeff_food_eaten = 10
        penality_died = 15
        coeff_time_taking_food = 0.1

        # if snek died
        if self.done:
            return self.prev_reward - penality_died

        reward += (1/self.calculate_dist_to_food()) * coeff_dist
        reward += self.food_eaten() * coeff_food_eaten

        reward = reward - self.prev_reward  # because the two above works with deltas
        self.prev_reward = reward

        reward -= coeff_time_taking_food / len(self.snakeGame.snake)

        return reward


    def choose_direction(self, action):
        # action = 0 : stay
        # action = 1 : left
        # action = 2 : right
        if self.snakeGame.next_direction == DIRECTION_UP:
            if action == 0:
                return DIRECTION_UP
            elif action == 1:
                return DIRECTION_LEFT
            else:
                return DIRECTION_RIGHT

        elif self.snakeGame.next_direction == DIRECTION_DOWN:
            if action == 0:
                return DIRECTION_DOWN
            elif action == 1:
                return DIRECTION_RIGHT
            else:
                return DIRECTION_LEFT

        elif self.snakeGame.next_direction == DIRECTION_RIGHT:
            if action == 0:
                return DIRECTION_RIGHT
            elif action == 1:
                return DIRECTION_UP
            else:
                return DIRECTION_DOWN

        else:
            if action == 0:
                return DIRECTION_LEFT
            elif action == 1:
                return DIRECTION_DOWN
            else:
                return DIRECTION_UP


    def step(self, action):
        # drawing
        self.snakeGame.draw()
        pygame.display.flip()

        # make the snake advance
        self.snakeGame.next_direction = self.choose_direction(action)
        self.snakeGame.update_gym()
        if not self.snakeGame.playing:
            self.done = True

        # take infos of the step
        observation = self.makeObservation()

        reward = self.define_reward()
        info = {}
        return observation, reward, self.done, info


    def reset(self):

        self.snakeGame.reset()
        self.snakeGame.draw()
        pygame.display.flip()
        self.prev_reward = 0
        self.previous_length = SNAKE_START_LENGTH
        self.done = False


        observation = self.makeObservation()

        return observation  # reward, done, info can't be included


    def render(self, mode='human'):
        time.sleep(0.1)


    def close(self):
        pygame.quit()
