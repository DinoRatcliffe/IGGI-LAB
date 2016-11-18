import copy
import random
import numpy as np

class GridWorld:
    def __init__(self, size):
        self.map = []
        for i in range(0, size):
            self.map.append([0] * size)

        self.player = {'x': random.randint(0, size-1), 'y': random.randint(0, size-1)}
        self.goal = {'x': random.randint(0, size-1), 'y': random.randint(0, size-1)}
        self.actions_taken = 0
        
        while self.goal['x'] == self.player['x'] and self.goal['y'] == self.player['y']:
            self.player = {'x': random.randint(0, size-1), 'y': random.randint(0, size-1)}

        self.GOAL = 255
        self.PLAYER = 128
        self.size = size
        self.goal_reached = False

    def min_remaining_moves(self):
        x_diff = abs(self.player['x'] - self.goal['x'])
        y_diff = abs(self.player['y'] - self.goal['y'])
        return x_diff + y_diff

        def actions_taken(self):
            return self.actions_taken

    def perform_action(self, action):
        self.actions_taken += 1
        actions = [self.move_up,
               self.move_right,
               self.move_down,
               self.move_left]
        actions[action]()

    def move_up(self):
        if self.player['y'] > 0:
            self.player['y'] -= 1

    def move_down(self):
        if self.player['y'] < self.size-1:
            self.player['y'] += 1
    
    def move_right(self):
        if self.player['x'] < self.size-1:
            self.player['x'] += 1

    def move_left(self):
        if self.player['x'] > 0:
            self.player['x'] -= 1

    def generate_states(self):
        states = []
        for i in range(self.size * self.size):
            for j in range(self.size * self.size):
                for k in range(self.size * self.size):
                    if not (i == j or k == j):
                        this_state = np.zeros((self.size, self.size))
                        this_state[i//self.size][i % self.size] = self.PLAYER
                        this_state[j//self.size][j % self.size] = self.GOAL

                        if i//self.size == k//self.size and i % self.size == k % self.size:
                            this_state[k//self.size][k%self.size] = self.PLAYER_PICKUP
                        else:
                            this_state[k//self.size][k%self.size] = self.PICKUP


                        states.append(this_state)
        return states

    def get_state(self):
        terminal = self.player['y'] == self.goal['y'] and self.player['x'] == self.goal['x'] 
        if terminal:
            self.goal_reached = True
        reward = 1 if self.player['y'] == self.goal['y'] and self.player['x'] == self.goal['x'] else 0 
                #reward -= 0.01
        state = [copy.copy(i) for i in self.map]
        state[self.goal['y']][self.goal['x']] = self.GOAL
        state[self.player['y']][self.player['x']] = self.PLAYER

        if self.actions_taken == 50:
            terminal = True
        return state, reward, terminal
