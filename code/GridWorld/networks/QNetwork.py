import tensorflow as tf
import numpy as np

class QNetwork:
    def __init__(self, num_actions, input_size):
        self.num_actions = num_actions
        self.input_size = input_size
        self.network_ops = self.create_network()

    def create_network(self):
        return {}
        ''' TODO create network
        return {'input': input_data,
                'output': output,
                'target': target,
                'action': action,
                'training_op': training_op}
        '''

    @staticmethod 
    def weight_variable(shape):
        """ Creates tensorflow variables in a specified shape to be used for weights 
        Args:
        shape: a vector indicating the shape of the resulting tensor 2x2 being:
        [2, 2]
        Returns:
        A tensor of the defined shape with small random values
        """
        initial = tf.truncated_normal(shape, stddev=1e-6)
        return tf.Variable(initial)

    @staticmethod
    def bias_variable(shape):
        """ Creates tensorflow variables in a specified shape to be used for bias 
        Args:
        shape: a vector indicating the shape of the resulting tensor 2x2 being:
        [2, 2]
        Returns:
        A tensor of the defined shape
        """
        initial = tf.constant(1e-6, shape=shape)
        return tf.Variable(initial)
