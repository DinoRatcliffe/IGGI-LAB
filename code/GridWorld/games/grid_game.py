from game import Game, FrameQueue
from gridworld import GridWorld
import numpy as np

class GridWrapper(Game):
    def __init__(self, size, frame_stack_size=1, render=False):
        self.size = size
        self.new_game()

    def possible_actions(self):
        num_actions = 4
        actions = []
        for i in range(num_actions):
            new_action = [0] * num_actions
            new_action[i] = 1
            actions.append(new_action)
        return actions

    def perform_action(self, action):
        action_idx = np.argmax(action)
        self.game.perform_action(action_idx)

    def get_state(self):
        state, reward, terminal = self.game.get_state()
        for i in range(len(state)):
            for j in range(len(state[i])):
                state[i][j] = [state[i][j]]
        return state, reward, terminal 

    def get_score(self):
        _, _, terminal = self.game.get_state()
        score = 0
        if terminal:
            score += 1
        score -= self.game.actions_taken * 0
        return score
    
    def goal_reached(self):
        return self.game.goal_reached

    def actions_taken(self):
        return self.game.actions_taken

    def new_game(self):
        self.game = GridWorld(self.size)
        self.min_moves = self.game.min_remaining_moves()

    def generate_states(self):
        return self.game.generate_states()
