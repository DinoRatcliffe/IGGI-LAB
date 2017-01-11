import numpy as np
from collections import deque

class TextBatcher():

    def __init__(self, text, batch_size, sequence_length):
        self.text = text
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.vocab = list(set(text))
        self.current_batch = 0

    def reset(self):
        self.current_batch = 0

    def convert_string(self, string):
        ids = []
        for char in string:
            ids.append(self._char_to_id(char))
        return ids
    def convert_ids(self, ids):
        string = ""
        for id_ in ids:
            string += self._id_to_char(id_)
        return string

    def has_next_batch(self):
        batch_start = self.current_batch * self.batch_size
        batch_end = batch_start + self.batch_size + self.sequence_length
        return batch_end < len(self.text)

    def next_batch(self):
        inputQue = deque(maxlen=self.sequence_length)
        targetQue = deque(maxlen=self.sequence_length)

        inputs = []
        targets = []

        batch_start = self.current_batch * self.batch_size
        batch_end = batch_start + self.batch_size + self.sequence_length

        for i in range(batch_start, batch_end):
            char_id = self._char_to_id(self.text[i])

            if len(targetQue) > 0:
                inputQue.append(targetQue[-1])

            targetQue.append(char_id)

            if len(inputQue) == self.sequence_length:
                inputs.append(np.array(inputQue))
                targets.append(np.array(targetQue))

        self.current_batch += 1

        return (inputs, targets)

    def vocab_ids(self):
        return [i for i in range(0, len(self.vocab))]

    def _char_to_id(self, char):
        if char in self.vocab:
            return self.vocab.index(char)
        else:
            print("not found: " + char)
            return len(self.vocab) - 1

    def _id_to_char(self, integer):
        if integer > len(self.vocab):
            return " "
        else: 
            return self.vocab[integer]
