import numpy as np
from pickle import load

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import (LSTM, Dense, Embedding, Input, RepeatVector,
                          TimeDistributed)
# from keras.utils.vis_utils import plot_model
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from numpy import array

from hdf5DatasetGenerator import HDF5DatasetGenerator
print("GPU Available: ", tf.test.is_gpu_available())

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    except RuntimeError as e:
        print(e)

class Token:
    def __init__(self, length, tokenizer):
        self.length = length
        self.tokenizer = tokenizer
        self.token = tokenizer.word_index.items()
    
    def startid():
        return self.tokenizer.word_index['bstart']
    
    def endid():
        return self.tokenizer.word_index['kend']


# load a clean dataset
def load_clean_sentences(filename):
    return load(open(filename, 'rb'))

# fit a tokenizer
def create_tokenizer(lines):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer

# max sentence length
def max_length(lines):
    return max(len(line.split()) for line in lines)

def encode_output(sequences, vocab_size):
    ylist = list()
    for sequence in sequences:
        encoded = to_categorical(sequence, num_classes=vocab_size)
        ylist.append(encoded)
        y = array(ylist)
    y = y.reshape(sequences.shape[0], sequences.shape[1], vocab_size)
    return y

def next_batch(data_path, batch_size, eng_length):
    generator = HDF5DatasetGenerator(dbPath=data_path, batchSize=batch_size, binarize=False)
    while True:
        generate = generator.generator()
        for X_data, Y_data, out_data in generate:
          inputs =[X_data, Y_data]
          outputs = encode_output(out_data, eng_vocab_size)
          yield (inputs, outputs)

# load datasets
dataset = load_clean_sentences('english-german-both.pkl')

# prepare english tokenizer
eng_tokenizer = create_tokenizer(dataset[:, 0])
eng_vocab_size = len(eng_tokenizer.word_index) + 1
eng_length = max_length(dataset[:, 0])
print('English Vocabulary Size: %d' % eng_vocab_size)
print('English Max Length: %d' % (eng_length))

# prepare german tokenizer
ger_tokenizer = create_tokenizer(dataset[:, 1])
ger_vocab_size = len(ger_tokenizer.word_index) + 1
ger_length = max_length(dataset[:, 1])
print('German Vocabulary Size: %d' % ger_vocab_size)
print('German Max Length: %d' % (ger_length))

# define model
s2s = Transformer(itokens, otokens, len_limit=70, d_model=d_model, d_inner_hid=512, \
				   n_head=8, layers=2, dropout=0.1)