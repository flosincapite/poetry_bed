"""
Code adapted from Adventures in Machine Learning [1] [2].

[1] https://adventuresinmachinelearning.com/word2vec-keras-tutorial/
[2] https://github.com/adventuresinML/adventures-in-ml-code/blob/master/keras_word2vec.py
"""

# TODO: Save symbol tables, models.
# TODO: Add load functionality.
# TODO: Put on GitHub.
# TODO: Figure out how to get actual embedding vector values--make embedding
#   an output.

import collections
import os
import urllib
import zipfile

from util import symbol_table

# TODO: Switch to TensorFlow API after finding an equivalent to merge.
from keras.models import Model
from keras.layers import Input, Dense, Reshape, dot, Dot
from keras.layers.embeddings import Embedding
from keras.preprocessing.sequence import skipgrams
from keras.preprocessing import sequence
import numpy as np
import requests
import tensorflow as tf


class DataReader:

    CHUNK_SIZE = 1000
    DataSet = collections.namedtuple(
            'DataSet', ['word_target', 'word_context', 'labels'])

    def __init__(self, file_name):
        self._file_name = file_name
        self._file = None
        self._buffer = collections.deque()

    def _close(self):
        if self._file is not None:
            self._file.close()

    def __del__(self):
        self._close()

    def reset(self):
        self._close()
        self._buffer.clear()
        self._file = open(self._file_name, 'r')

    def read_all_words(self):
        self.reset()
        return self.read_words(until_end=True)

    def read_words(self, number_words=None, until_end=False):
        if self._file is None or self._file.closed:
            raise Exception('Nope')

        yielded = 0

        while True:
            while len(self._buffer) > 1:
                if number_words is None or yielded < number_words:
                    yield self._buffer.popleft()
                    yielded += 1
                else:
                    return
                    
            chunk = self._file.read(self.CHUNK_SIZE)
            if self._buffer:
                full_chunk = self._buffer.pop() + chunk
            else:
                full_chunk = chunk

            if not chunk:
                if until_end:
                    while self._buffer:
                        yield self._buffer.popleft()
                    self._close()
                    return
                else:
                    self.reset()

            self._buffer.extend(full_chunk.split())

    def get_data(self, data, vocab_size, window_size):
        sampling_table = sequence.make_sampling_table(vocab_size)
        couples, labels = skipgrams(
                data, vocab_size, window_size=window_size,
                sampling_table=sampling_table)
        word_target, word_context = zip(*couples)
        word_target = np.array(word_target, dtype="int32")
        word_context = np.array(word_context, dtype="int32")
        return self.DataSet(word_target, word_context, labels)


# TODO: Determine vocabulary size information-theoretically.
VOCAB_SIZE = 10000
FILENAME = 'text8'
WINDOW_SIZE = 3
VECTOR_DIM = 300
NUMBER_WORDS = 50000
EPOCHS = 200000


reader = DataReader(FILENAME)
symbols = symbol_table.SymbolTable(VOCAB_SIZE)
symbols.populate(reader)


reader.reset()
print(list(reader.read_words(number_words=7))[:7])
reader.reset()


valid_size = 1     # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
valid_examples = np.random.choice(valid_window, valid_size, replace=False)

# create some input variables
input_target = Input((1,))
input_context = Input((1,))

embedding = Embedding(VOCAB_SIZE, VECTOR_DIM, input_length=1, name='embedding')
target = embedding(input_target)
target = Reshape((VECTOR_DIM, 1))(target)
context = embedding(input_context)
context = Reshape((VECTOR_DIM, 1))(context)

# setup a cosine similarity operation which will be output in a secondary model
similarity = Dot(axes=(1, 1), normalize=True)([target, context])

# now perform the dot product operation to get a similarity measure
dot_product = Dot(axes=(1, 1), normalize=False)([target, context])
dot_product = Reshape((1,))(dot_product)
# add the sigmoid output layer
output = Dense(1, activation='sigmoid', name='output')(dot_product)
# create the primary training model
model = Model(inputs=[input_target, input_context], outputs=[target, output])
model.compile(loss={'output': 'binary_crossentropy'}, optimizer='rmsprop')

# create a secondary validation model to run our similarity checks during training
validation_model = Model(inputs=[input_target, input_context], outputs=[similarity])


class SimilarityCallback:
    def run_sim(self):
        for i in range(valid_size):
            valid_word = symbols.reversed_dictionary[valid_examples[i]]
            top_k = 8  # number of nearest neighbors
            sim = self._get_sim(valid_examples[i])
            nearest = (-sim).argsort()[1:top_k + 1]
            log_str = 'Nearest to %s:' % valid_word
            for k in range(top_k):
                close_word = symbols.reversed_dictionary[nearest[k]]
                log_str = '%s %s,' % (log_str, close_word)
            print(log_str)

    @staticmethod
    def _get_sim(valid_word_idx):
        sim = np.zeros((VOCAB_SIZE,))
        in_arr1 = np.zeros((1,))
        in_arr2 = np.zeros((1,))
        in_arr1[0,] = valid_word_idx
        for i in range(VOCAB_SIZE):
            in_arr2[0,] = i
            out = validation_model.predict_on_batch([in_arr1, in_arr2])
            sim[i] = out
        return sim
sim_cb = SimilarityCallback()

arr_1 = np.zeros((1,))
arr_2 = np.zeros((1,))
arr_3 = np.zeros((1,))

try:
    for cnt in range(EPOCHS):
        if cnt % 100 == 0:
            words = symbols.words_to_indices(reader.read_words(NUMBER_WORDS))
            word_target, word_context, labels = reader.get_data(
                    words, VOCAB_SIZE, WINDOW_SIZE)
        idx = np.random.randint(0, len(labels)-1)
        arr_1[0,] = word_target[idx]
        arr_2[0,] = word_context[idx]
        arr_3[0,] = labels[idx]
        loss = model.train_on_batch([arr_1, arr_2], arr_3)
        if cnt % 100 == 0:
            print("Iteration {}, loss={}".format(cnt, loss))
        if cnt % 10000 == 0:
            sim_cb.run_sim()
finally:
    print('Saving model.')
    model.save('trained_model')
    print('Saving symbol table.')
    symbols.write('syms.txt')
