import os
from pickle import load
from random import randint

import numpy as np

import tensorflow as tf
from numpy import argmax, array, array_equal
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import (GRU, LSTM, Dense, Embedding, Input,
                                     RepeatVector, TimeDistributed,
                                     concatenate)
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical

from bahdanauAttention import AttentionLayer
from hdf5DatasetGenerator import HDF5DatasetGenerator

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

# with tf.device('/CPU:0'):
# define model
hidden_size = 256
encoder_inputs = Input(shape=(ger_length,))
encoder_embedding = Embedding(ger_vocab_size, 256 , mask_zero=True, name='encoder_embedding') (encoder_inputs)
decoder_inputs = Input(shape=(eng_length,))
decoder_embedding = Embedding(eng_vocab_size, 256 , mask_zero=True, name='decoder_embedding') (decoder_inputs)

# Encoder GRU
encoder_gru = GRU(256, return_sequences=True, return_state=True, name='encoder_gru')
encoder_out, encoder_state = encoder_gru(encoder_embedding)

# Set up the decoder GRU, using `encoder_states` as initial state.
decoder_gru = GRU(256, return_sequences=True, return_state=True, name='decoder_gru')
decoder_out, decoder_state = decoder_gru(decoder_embedding, initial_state=encoder_state)

# Attention layer
attn_layer = AttentionLayer(name='attention_layer')
attn_out, attn_states = attn_layer([encoder_out, decoder_out])

# Concat attention input and decoder GRU output
decoder_concat_input = concatenate(inputs=[decoder_out, attn_out], axis=-1, name='concat_layer')

# Dense layer
dense = Dense(eng_vocab_size, activation='softmax', name='softmax_layer')
dense_time = TimeDistributed(dense, name='time_distributed_layer')
decoder_pred = dense_time(decoder_concat_input)

# Full model
full_model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=decoder_pred)
full_model.compile(optimizer='adam', loss='categorical_crossentropy')

full_model.summary()

batch_size = 128
checkpoint = ModelCheckpoint('model.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
full_model.fit_generator(
    generator = next_batch('translate_train.hdf5', batch_size, eng_length), 
    steps_per_epoch = 1000,
    epochs=25,
    validation_data=next_batch('translate_test.hdf5', batch_size, eng_length), 
    validation_steps = 200,
    callbacks=[checkpoint]
)

""" Encoder (Inference) model """
encoder_inf_inputs = Input(shape=(ger_length,), name='encoder_inf_inputs')
encoder_inf_embedding = Embedding(ger_vocab_size, 256 , mask_zero=True, name='encoder_inf_embedding') (encoder_inf_inputs)
encoder_inf_out, encoder_inf_state = encoder_gru(encoder_inf_embedding)
encoder_model = Model(inputs=encoder_inf_inputs, outputs=[encoder_inf_out, encoder_inf_state])
encoder_model.summary()

""" Decoder (Inference) model """
decoder_inf_inputs = Input(shape=(1, ), name='decoder_word_inputs')
decoder_inf_embedding = Embedding(eng_vocab_size, 256 , mask_zero=True ) (decoder_inf_inputs)
encoder_inf_states = Input(shape=(ger_length, hidden_size), name='encoder_inf_states')
decoder_init_state = Input(shape=(hidden_size,), name='decoder_init')

decoder_inf_out, decoder_inf_state = decoder_gru(decoder_inf_embedding, initial_state=decoder_init_state)
attn_inf_out, attn_inf_states = attn_layer([encoder_inf_states, decoder_inf_out])
decoder_inf_concat = concatenate(inputs=[decoder_inf_out, attn_inf_out], axis=-1, name='concat')
decoder_inf_pred = TimeDistributed(dense)(decoder_inf_concat)
decoder_model = Model(inputs=[encoder_inf_states, decoder_init_state, decoder_inf_inputs],
                      outputs=[decoder_inf_pred, attn_inf_states, decoder_inf_state])


def encode_sequences(tokenizer, length, text):
    # integer encode sequences
    X = tokenizer.texts_to_sequences([text])
    # pad sequences with 0 values
    X = pad_sequences(X, maxlen=length, padding='post')
    return X

en_index2word = dict(zip(eng_tokenizer.word_index.values(), eng_tokenizer.word_index.keys()))

while True:
    enc_outs, enc_last_state = encoder_model.predict( encode_sequences(ger_tokenizer, ger_length, input( 'Enter german sentence : ')) )
    eng_seq = np.zeros( ( 1 , 1 ) )
    eng_seq[0, 0] = eng_tokenizer.word_index['bstart']
    dec_state = enc_last_state
    attention_weights = []
    eng_text = ''
    stop_condition = False
    print('eng_seq.shape ', eng_seq.shape)
    print('enc_outs.shape ', enc_outs.shape)
    print('enc_last_state.shape ', enc_last_state.shape)
    while not stop_condition:
        dec_out, attention, dec_state = decoder_model.predict([enc_outs, dec_state, eng_seq])
        dec_ind = np.argmax(dec_out, axis=-1)[0, 0]
        sampled_word = en_index2word[dec_ind]
        eng_seq = np.zeros( ( 1 , 1 ) )
        eng_seq[0, 0] = dec_ind
        if sampled_word == 'kend' or len(eng_text.split()) > eng_length:
            stop_condition = True
        
        attention_weights.append((dec_ind, attention))
        eng_text +=  sampled_word + ' '
    print('eng_text ', eng_text)
