import re
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

def text_to_vietkey(s):
    s = s.lower()
    s = re.sub( 'bstart ' , '', s)
    s = re.sub( 'dend ' , '', s)
    s = re.sub( 'aws', 'ắ', s)
    s = re.sub( 'awf', 'ằ', s)
    s = re.sub( 'awr', 'ẳ', s)
    s = re.sub( 'awx', 'ẵ', s)
    s = re.sub( 'aar', 'ẩ', s)
    s = re.sub( 'aas', 'ấ', s)
    s = re.sub( 'aax', 'ẫ', s)
    s = re.sub( 'oof', 'ồ', s)
    s = re.sub( 'uws', 'ứ', s)
    s = re.sub( 'uwf', 'ừ', s)
    s = re.sub( 'uwr', 'ử', s)
    s = re.sub( 'uwx', 'ữ', s)
    s = re.sub( 'uwj', 'ự', s)
    s = re.sub( 'oox', 'ỗ', s)
    s = re.sub( 'ooj', 'ộ', s)
    s = re.sub( 'aaf', 'ầ', s)
    s = re.sub( 'aaj', 'ậ', s)
    s = re.sub( 'awj', 'ặ', s)
    s = re.sub( 'eef', 'ề', s)
    s = re.sub( 'ees', 'ế', s)
    s = re.sub( 'oos', 'ố', s)
    s = re.sub( 'eex', 'ễ', s)
    s = re.sub( 'oor', 'ổ', s)
    s = re.sub( 'eer', 'ể', s)
    s = re.sub( 'eej', 'ệ', s)
    s = re.sub( 'ows', 'ớ', s)
    s = re.sub( 'owf', 'ờ', s)
    s = re.sub( 'owr', 'ở', s)
    s = re.sub( 'owx', 'ỡ', s)
    s = re.sub( 'owj', 'ợ', s)
    s = re.sub( 'is' , 'í', s)
    s = re.sub( 'as' , 'á', s)
    s = re.sub( 'af' , 'à', s)
    s = re.sub( 'ar' , 'ả', s)
    s = re.sub( 'ax' , 'ã', s)
    s = re.sub( 'aj' , 'ạ', s)
    s = re.sub( 'aw' , 'ă', s)
    s = re.sub( 'aa' , 'â', s)
    s = re.sub( 'es' , 'é', s)
    s = re.sub( 'ef' , 'è', s)
    s = re.sub( 'er' , 'ẻ', s)
    s = re.sub( 'ex' , 'ẽ', s)
    s = re.sub( 'ej' , 'ẹ', s)
    s = re.sub( 'ee' , 'ê', s)
    s = re.sub( 'os' , 'ó', s)
    s = re.sub( 'of' , 'ò', s)
    s = re.sub( 'or' , 'ỏ', s)
    s = re.sub( 'ox' , 'õ', s)
    s = re.sub( 'oj' , 'ọ', s)
    s = re.sub( 'oo' , 'ô', s)
    s = re.sub( 'ow' , 'ơ', s)
    s = re.sub( 'if' , 'ì', s)
    s = re.sub( 'ir' , 'ỉ', s)
    s = re.sub( 'ix' , 'ĩ', s)
    s = re.sub( 'ij' , 'ị', s)
    s = re.sub( 'us' , 'ú', s)
    s = re.sub( 'uf' , 'ù', s)
    s = re.sub( 'ur' , 'ủ', s)
    s = re.sub( 'ux' , 'ũ', s)
    s = re.sub( 'uj' , 'ụ', s)
    s = re.sub( 'uw' , 'ư', s)
    s = re.sub( 'ys' , 'ý', s)
    s = re.sub( 'yf' , 'ỳ', s)
    s = re.sub( 'yr' , 'ỷ', s)
    s = re.sub( 'yx' , 'ỹ', s)
    s = re.sub( 'yj' , 'ỵ', s)
    s = re.sub( 'dd' , 'đ', s)
    return s

# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     try:
#         tf.config.experimental.set_visible_devices(gpus[2], 'GPU')
#         logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#         print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
#     except RuntimeError as e:
#         print(e)

# load a clean dataset
def load_clean_sentences(filename):
    return load(open(filename, 'rb'))

# fit a tokenizer
def create_tokenizer(lines):
    tokenizer = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\\]^`{|}~\t\n')
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
dataset = load_clean_sentences('data/english-vietnamese-both-all.pkl')

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
print("data ", eng_tokenizer.word_index.items())

# # define model
# encoder_inputs = Input(shape=( None , ))
# encoder_embedding = Embedding( ger_vocab_size, 256 , mask_zero=True ) (encoder_inputs)
# encoder_outputs , state_h , state_c = LSTM( 256 , return_state=True , recurrent_dropout=0.2 , dropout=0.2 )( encoder_embedding )
# encoder_states = [ state_h , state_c ]

# decoder_inputs = Input(shape=( None ,  ))
# decoder_embedding = Embedding( eng_vocab_size, 256 , mask_zero=True) (decoder_inputs)
# decoder_lstm = LSTM( 256 , return_state=True , return_sequences=True , recurrent_dropout=0.2 , dropout=0.2)
# decoder_outputs , _ , _ = decoder_lstm ( decoder_embedding , initial_state=encoder_states )
# decoder_dense = Dense( eng_vocab_size , activation="softmax" ) 
# output = decoder_dense ( decoder_outputs )

# model = Model([encoder_inputs, decoder_inputs], output )
# model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['acc'])

# model.summary()
# try:
#     model.load_weights( 'model/model.h5' )
# except:
#     print('no model')
# # fit model
# checkpoint = ModelCheckpoint('model/model.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
# batch_size = 128
# model.fit_generator(
#     generator = next_batch('hdf5/translate_train_all.hdf5', batch_size, eng_length), 
#     steps_per_epoch = int(32174 / 128),
#     epochs=25,
#     validation_data=next_batch('hdf5/translate_test_all.hdf5', batch_size, eng_length), 
#     validation_steps = int(8044 / 128),
#     callbacks=[checkpoint]
# )

# def make_inference_models():
    
#     encoder_model = Model(encoder_inputs, encoder_states)
    
#     decoder_state_input_h = Input(shape=( 256,))
#     decoder_state_input_c = Input(shape=( 256 ,))
    
#     decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    
#     decoder_outputs, state_h, state_c = decoder_lstm(
#         decoder_embedding , initial_state=decoder_states_inputs)
#     decoder_states = [state_h, state_c]
#     decoder_outputs = decoder_dense(decoder_outputs)
#     decoder_model = Model(
#         [decoder_inputs] + decoder_states_inputs,
#         [decoder_outputs] + decoder_states)
    
#     return encoder_model , decoder_model

# def encode_sequences(tokenizer, length, text):
#     # integer encode sequences
#     X = tokenizer.texts_to_sequences([text])
#     # pad sequences with 0 values
#     X = pad_sequences(X, maxlen=length, padding='post')
#     return X

# enc_model , dec_model = make_inference_models()

# # load datasets
# dataset = load_clean_sentences('data/english-vietnamese-both-all.pkl')
# test_data = load_clean_sentences('data/english-vietnamese-test-all.pkl')

# # for text in test_data:
# #     raw_target, raw_src = text 
# #     states_values = enc_model.predict( encode_sequences(ger_tokenizer, ger_length, raw_src) )
# #     #states_values = enc_model.predict( encoder_input_data[ epoch ] )
# #     empty_target_seq = np.zeros( ( 1 , 1 ) )
# #     empty_target_seq[0, 0] = eng_tokenizer.word_index['bstart']
# #     stop_condition = False
# #     a = [ empty_target_seq ] + states_values
# #     decoded_translation = ''
# #     while not stop_condition :
# #         dec_outputs , h , c = dec_model.predict([ empty_target_seq ] + states_values )
# #         sampled_word_index = np.argmax( dec_outputs[0, -1, :] )
# #         sampled_word = None
# #         for word , index in eng_tokenizer.word_index.items() :
# #             if sampled_word_index == index :
# #                 decoded_translation += ' {}'.format( word )
# #                 sampled_word = word
        
# #         if sampled_word == 'kend' or len(decoded_translation.split()) > eng_length:
# #             stop_condition = True
            
# #         empty_target_seq = np.zeros( ( 1 , 1 ) )  
# #         empty_target_seq[ 0 , 0 ] = sampled_word_index
# #         states_values = [ h , c ] 
# #     print('src=[%s], target=[%s], predicted=[%s]' % (raw_src, text_to_vietkey(raw_target), text_to_vietkey(decoded_translation)))

# while True:
#     states_values = enc_model.predict( encode_sequences(ger_tokenizer, ger_length, input( 'Enter german sentence : ')) )
#     #states_values = enc_model.predict( encoder_input_data[ epoch ] )
#     empty_target_seq = np.zeros( ( 1 , 1 ) )
#     empty_target_seq[0, 0] = eng_tokenizer.word_index['bstart']
#     stop_condition = False
#     a = [ empty_target_seq ] + states_values
#     decoded_translation = ''
#     while not stop_condition :
#         dec_outputs , h , c = dec_model.predict([ empty_target_seq ] + states_values )
#         sampled_word_index = np.argmax( dec_outputs[0, -1, :] )
#         sampled_word = None
#         for word , index in eng_tokenizer.word_index.items() :
#             if sampled_word_index == index :
#                 decoded_translation += ' {}'.format( word )
#                 sampled_word = word
        
#         if sampled_word == 'kend' or len(decoded_translation.split()) > eng_length:
#             stop_condition = True
            
#         empty_target_seq = np.zeros( ( 1 , 1 ) )  
#         empty_target_seq[ 0 , 0 ] = sampled_word_index
#         states_values = [ h , c ] 

#     print( decoded_translation )

# # def evaluate_model(model, sources, raw_dataset):
# #     actual, predicted = list(), list()
# #     for i, source in enumerate(sources):
# #         # translate encoded source text
# #         source = source.reshape((1, source.shape[0]))
# #         translation = predict_sequence(model, eng_tokenizer, source)
# #         raw_target, raw_src = raw_dataset[i]
# #         if i < 100:
# #             print('src=[%s], target=[%s], predicted=[%s]' % (raw_src, raw_target, translation))
# #         actual.append(raw_target.split())
# #         predicted.append(translation.split())
# #     # calculate BLEU score
# #     print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
# #     print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
# #     print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
# #     print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))


