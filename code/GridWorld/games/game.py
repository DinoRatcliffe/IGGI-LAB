from collections import deque
import numpy as np

class Game(object):
    def __init__(self):
        raise NotImplementedError

    def perform_action(self, action):
        raise NotImplementedError

    def get_state(self):
        raise NotImplementedError

    def get_score(self):
        raise NotImplementedError

    def new_game(self):
        raise NotImplementedError

class FrameQueue:
    """ A class that holds the last n frames of the game state 
    Attributes:
        size: the number of frames to be stored 
        memory: A deque that stores the frames 
    """

    def __init__(self, size, channels_per_image = 3):
        """ Initialize the memory
        """
        self.size = size
        self.memory = deque(maxlen=size)
        self.channels_per_image = channels_per_image
        self.zipped = []

    def add_frame(self, frame):
        """ Adds a frame to memory
        Args:
            frame: a 2D vector of the game frame
        """
        self.memory.append(frame)

    def filled(self):
        return len(self.memory) == self.size

    def zip(self):
        return np.moveaxis(np.array(self.memory), 1, 3)[0]

