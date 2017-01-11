import tensorflow as tf
import numpy as np
from collections import deque
import os

class LSTMNetwork():
    def __init__(self, config, vocab_length):
        #TODO(yo@dino.io): allow config to be passed in
        self.model_dir = config['model_dir']
        self.seq_length = config['seq_length'] 
        self.batch_size = config['batch_size']
        self.hidden_size = config['hidden_size']
        self.num_layers = config['num_layers']
        self.keep_prob = config['keep_prob']
        self.gradient_clip = config['gradient_clip']
        self.learning_rate = config['learning_rate']
        self.training = config['training']
        self.init_scale = config['init_scale']
        self.vocab_length = vocab_length

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

    def save_model(self, t_step):
        self.saver.save(self.sess, self.model_dir + '/model.ckpt', t_step)

    def start_session(self):
        self.net = self.create_network();
        self.saver = tf.train.Saver();
        self.sess = tf.Session()

        self.summaries = tf.merge_all_summaries()
        self.train_wirter = tf.train.SummaryWriter('logs/train', self.sess.graph)
        self.summary_wirter = tf.train.SummaryWriter('logs/summary')

        initializer = tf.random_uniform_initializer(-self.init_scale, self.init_scale)
        self.sess.run(tf.initialize_all_variables())

        ckpt = tf.train.get_checkpoint_state(self.model_dir)

        #TODO(yo@dino.io): do logging instead of print
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            print('loaded saved model at: ' + self.model_dir)
        else:
            print('No model found at: ' + self.model_dir)

    def end_session(self):
        self.sess.close()

    def create_network(self):
        input_data = tf.placeholder(tf.int32, [self.batch_size, self.seq_length])
        targets = tf.placeholder(tf.int32, [self.batch_size, self.seq_length])

        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size, forget_bias=1.0, state_is_tuple=True)
        if self.training:
            lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=self.keep_prob)

        cells = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * self.num_layers, state_is_tuple=True)

        initial_state = cells.zero_state(self.batch_size, tf.float32)
        state = initial_state

        with tf.device("/cpu:0"):
            embedding = tf.get_variable("embedding", [self.vocab_length, self.hidden_size])
            input_embeddings = tf.nn.embedding_lookup(embedding, input_data)

        if self.training:
            input_embeddings = tf.nn.dropout(input_embeddings, self.keep_prob)

        # converts inputs to list of [batch, seq_length]
        input_embeddings = [tf.squeeze(single_input, [1]) 
                for single_input in tf.split(1, self.seq_length, input_embeddings)]

        lstm_output, state = tf.nn.rnn(cells, input_embeddings, initial_state=initial_state)
        lstm_state = state

        #combine list
        lstm_output_reshape = tf.reshape(tf.concat(1, lstm_output), [-1, self.hidden_size])

        W_softmax = tf.get_variable("W_softmax", [self.hidden_size, self.vocab_length])
        b_softmax = tf.get_variable("b_softmax", [self.vocab_length])
        h_softmax = tf.matmul(lstm_output_reshape, W_softmax) + b_softmax 

        loss = tf.nn.seq2seq.sequence_loss_by_example(
                [h_softmax],
                [tf.reshape(targets, [-1])],
                [tf.ones([self.batch_size * self.seq_length])])

        cost = tf.reduce_sum(loss) / self.batch_size

        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), self.gradient_clip)
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        train = optimizer.apply_gradients(zip(grads, tvars))
        return {'input': input_data,
                'output': h_softmax,
                'target': targets,
                'state': lstm_state,
                'initial_state': initial_state,
                'cost': cost,
                'training_op': train}

    def train(self, inputs, targets):

        feed_dict = {self.net['input']: inputs,
                 self.net['target']: targets}

        state = []
        for c, m in self.net['initial_state']: # initial_state: ((c1, m1), (c2, m2))
                state.append((self.sess.run(c), self.sess.run(m)))
        for i, (c, m) in enumerate(self.net['initial_state']):
                  feed_dict[c], feed_dict[m] = state[i]

        fetches = []
        fetches.append(self.net['cost'])
        fetches.append(self.net['training_op'])

        for c, m in self.net['state']:
            fetches.append(c)
            fetches.append(m)

        output = self.sess.run(fetches,
                feed_dict)

        cost = output[0]
        state_flat = output[2:]
        state = [state_flat[i:i+2] for i in range(0, len(state_flat), 2)]

        return (cost, state)

    def prime_network(self, prime_data):
        state = False
        for identifier in prime_data:
            output, state = self.infer(identifier, state=state)

        return (output, state)

    def infer(self, identifier, state=False):
        feed_dict = {self.net['input']: [[identifier]]}
        if type(state) == bool:
            state = []
            for c, m in self.net['initial_state']: # initial_state: ((c1, m1), (c2, m2))
                    state.append((self.sess.run(c), self.sess.run(m)))

        fetches = []
        fetches.append(self.net['output'])
        for c, m in self.net['state']:
            fetches.append(c)
            fetches.append(m)

        for i, (c, m) in enumerate(self.net['initial_state']):
            feed_dict[c], feed_dict[m] = state[i]

        eval_output = self.sess.run(fetches, feed_dict)
        output = eval_output[0]
        state_flat = eval_output[1:]
        state = [state_flat[i:i+2] for i in range(0, len(state_flat), 2)]

        return (output, state)

    def generate_text(self, prime_data, length):
        generated_output = []

        output, state = self.prime_network(prime_data)
        output_id = self._pick_item(output[0])
        generated_output.append(output_id)

        for i in range(0, length):
            output, state = self.infer(output_id, state=state)
            output_id = self._pick_item(output[0])

            generated_output.append(output_id)

        return generated_output

    def _pick_item(self, output):
        noise = np.random.normal(0, 1, len(output))
        return np.argmax(output + noise)
