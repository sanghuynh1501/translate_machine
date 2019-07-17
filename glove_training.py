import numpy as np
from pickle import load
from numpy import array
from mlcollect import HDF5DatasetGenerator
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
# from keras.utils.vis_utils import plot_model
from keras.models import Sequential, Model
from keras.layers import LSTM
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.callbacks import ModelCheckpoint
from keras.initializers import Constant

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

# define NMT model
def define_model(src_vocab, tar_vocab, src_timesteps, tar_timesteps, n_units, embedding_matrix):
    encoder_input = Input((src_timesteps,))
    encoder_embedding = Embedding(src_vocab, n_units, input_length=src_timesteps, mask_zero=True)(encoder_input)
    _, encoder_h, encoder_c = LSTM(256, return_state = True)(encoder_embedding)
    encoder_states = [encoder_h, encoder_c]
    
    decoder_input = Input((tar_timesteps,))
    decoder_embedding = Embedding(tar_vocab, n_units, embeddings_initializer=Constant(embedding_matrix), input_length=tar_timesteps, mask_zero=True)(decoder_input)
    decoder_LSTM = LSTM(256, return_sequences = True, return_state = True)
    decoder_out, _, _ = decoder_LSTM(decoder_embedding, initial_state=encoder_states)
    decoder_dense = Dense(tar_vocab, activation="softmax")
    decoder_out = decoder_dense (decoder_out)

    model = Model(inputs=[encoder_input, decoder_input], outputs=[decoder_out])
    model.summary()
    model.compile(optimizer="rmsprop", loss="categorical_crossentropy")
    return model

def next_batch(data_path, batch_size):
    generator = HDF5DatasetGenerator(dbPath=data_path, batchSize=batch_size, binarize=False)
    while True:
        generate = generator.generator()
        for X_data, Y_data, _ in generate:
            inputs =[X_data, Y_data]
            outputs = encode_output(Y_data, eng_vocab_size)
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

# Glover
word_index = eng_tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

embeddings_index = {}
with open('glove.6B.100d.txt') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

# prepare embedding matrix
num_words = min(eng_vocab_size, len(word_index)) + 1
embedding_matrix = np.zeros((num_words, 256))
for word, i in word_index.items():
    if i > eng_vocab_size:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

# define model
model = define_model(ger_vocab_size, eng_vocab_size, ger_length, eng_length, 256, embedding_matrix)
# fit model
checkpoint = ModelCheckpoint('model.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
batch_size = 64
model.fit_generator(
    generator = next_batch('translate_train.hdf5', batch_size), 
    steps_per_epoch = int(16000 / batch_size),
    epochs=500,
    validation_data=next_batch('translate_test.hdf5', batch_size), 
    validation_steps = int(4000 / batch_size),
    callbacks=[checkpoint]
)