from games.grid_game import GridWrapper
from networks.QNetwork import QNetwork
from collections import deque
import tensorflow as tf
import numpy as np

# params
epochs = 1000000
memory_size = 100000
memory_bootstrap_size = 100000
grid_size = 5
training_sample_size = 32

start_epsilon = 0.5
end_epsilon = 0.1
epsilon_degrade_steps = 1000000
current_epsilon = start_epsilon

# objects
experiance_memory = deque(maxlen=memory_size)
game = GridWrapper(grid_size)
actions = game.possible_actions()

network = QNetwork(len(actions), grid_size*grid_size)
network_ops = network.network_ops

# bootstrap memory with experiences 
for i in range(memory_bootstrap_size):
    state, reward, terminal = game.get_state()

    if terminal:
        game.new_game()
        state, reward, terminal = game.get_state()

    action = actions[np.random.choice(range(len(actions)))]
    game.perform_action(action)

    state_t1, reward, terminal = game.get_state()
    experiance_memory.append({'S'        : np.array(state).flatten(), 
                              'a'        : action, 
                              'R'        : reward, 
                              'S_t+1'    : np.array(state_t1).flatten(), 
                              'terminal' : terminal})

sess = tf.Session()
sess.run(tf.initialize_all_variables())

# training
for i in range(epochs):
    #TODO train on experiences
    #TODO generate new experience 
    if current_epsilon > end_epsilon:
        current_epsilon -= (start_epsilon - end_epsilon) / epsilon_degrade_steps

# helper functions
def pick_action(state, epsilon):
    #TODO pick action based on epsilon
    return actions[0]
