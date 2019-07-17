import re
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers , activations , models , preprocessing , utils
import pandas as pd

print( tf.VERSION )

import requests, zipfile, io    

def text_to_vietkey(s):
    s = s.lower()
    s = re.sub('á', 'as' , s)
    s = re.sub('à', 'af' , s)
    s = re.sub('ả', 'ar' , s)
    s = re.sub('ã', 'ax' , s)
    s = re.sub('ạ', 'aj' , s)
    s = re.sub('ă', 'aw' , s)
    s = re.sub('ắ', 'aws' , s)
    s = re.sub('ằ', 'awf' , s)
    s = re.sub('ẳ', 'awr' , s)
    s = re.sub('ẵ', 'awx' , s)
    s = re.sub('ặ', 'awj' , s)
    s = re.sub('â', 'aa' , s)
    s = re.sub('ấ', 'aas' , s)
    s = re.sub('ầ', 'aaf' , s)
    s = re.sub('ẩ', 'aar' , s)
    s = re.sub('ẫ', 'aax' , s)
    s = re.sub('ậ', 'aaj' , s)
    s = re.sub('é', 'es' , s)
    s = re.sub('è', 'ef' , s)
    s = re.sub('ẻ', 'er' , s)
    s = re.sub('ẽ', 'ex' , s)
    s = re.sub('ẹ', 'ej' , s)
    s = re.sub('ê', 'ee' , s)
    s = re.sub('ế', 'ees' , s)
    s = re.sub('ề', 'eef' , s)
    s = re.sub('ể', 'eer' , s)
    s = re.sub('ễ', 'eex' , s)
    s = re.sub('ệ', 'eej' , s)
    s = re.sub('ó', 'os' , s)
    s = re.sub('ò', 'of' , s)
    s = re.sub('ỏ', 'or' , s)
    s = re.sub('õ', 'ox' , s)
    s = re.sub('ọ', 'oj' , s)
    s = re.sub('ô', 'oo' , s)
    s = re.sub('ố', 'oos' , s)
    s = re.sub('ồ', 'oof' , s)
    s = re.sub('ổ', 'oor' , s)
    s = re.sub('ỗ', 'oox' , s)
    s = re.sub('ộ', 'ooj' , s)
    s = re.sub('ơ', 'ow' , s)
    s = re.sub('ớ', 'ows' , s)
    s = re.sub('ờ', 'owf' , s)
    s = re.sub('ở', 'owr' , s)
    s = re.sub('ỡ', 'owx' , s)
    s = re.sub('ợ', 'owj' , s)
    s = re.sub('í', 'is' , s)
    s = re.sub('ì', 'if' , s)
    s = re.sub('ỉ', 'ir' , s)
    s = re.sub('ĩ', 'ix' , s)
    s = re.sub('ị', 'ij' , s)
    s = re.sub('ú', 'us' , s)
    s = re.sub('ù', 'uf' , s)
    s = re.sub('ủ', 'ur' , s)
    s = re.sub('ũ', 'ux' , s)
    s = re.sub('ụ', 'uj' , s)
    s = re.sub('ư', 'uw' , s)
    s = re.sub('ứ', 'uws' , s)
    s = re.sub('ừ', 'uwf' , s)
    s = re.sub('ử', 'uwr' , s)
    s = re.sub('ữ', 'uwx' , s)
    s = re.sub('ự', 'uwj' , s)
    s = re.sub('ý', 'ys' , s)
    s = re.sub('ỳ', 'yf' , s)
    s = re.sub('ỷ', 'yr' , s)
    s = re.sub('ỹ', 'yx' , s)
    s = re.sub('ỵ', 'yj' , s)
    s = re.sub('đ', 'dd' , s)
    return s

file = open('vie.txt', mode='rt', encoding='utf-8')
text = file.read()
# text = text_to_vietkey(text)
file.close()
lines = text.strip().split('\n')
# r = requests.get( 'http://www.manythings.org/anki/vie-eng.zip' ) 
# z = zipfile.ZipFile(io.BytesIO('vie-eng.zip'))
# z.extractall() 
# lines = pd.read_table( 'vie.txt' , names=[ 'eng' , 'french' ] )
# lines = lines.iloc[ 10000 : 20000 ] 
# lines.head()
# eng_lines = list()
# for line in lines.eng:
#     eng_lines.append( line ) 
eng_lines = []
french_lines = []
for line in lines[:10000]:
  pairs = line.split('\t')
  eng_lines.append(pairs[1])
  french_lines.append( '<START> ' + pairs[0] + ' <END>' )

tokenizer = preprocessing.text.Tokenizer()
tokenizer.fit_on_texts( eng_lines ) 
tokenized_eng_lines = tokenizer.texts_to_sequences( eng_lines ) 

length_list = list()
for token_seq in tokenized_eng_lines:
    length_list.append( len( token_seq ))
max_input_length = np.array( length_list ).max()
print( 'English max length is {}'.format( max_input_length ))

padded_eng_lines = preprocessing.sequence.pad_sequences( tokenized_eng_lines , maxlen=max_input_length , padding='post' )
encoder_input_data = np.array( padded_eng_lines )
print( 'Encoder input data shape -> {}'.format( encoder_input_data.shape ))

eng_word_dict = tokenizer.word_index
num_eng_tokens = len( eng_word_dict )+1
print( 'Number of English tokens = {}'.format( num_eng_tokens))
# french_lines = list()
# for line in lines.french:
#     french_lines.append( '<START> ' + line + ' <END>' )  

tokenizer = preprocessing.text.Tokenizer()
tokenizer.fit_on_texts( french_lines ) 
tokenized_french_lines = tokenizer.texts_to_sequences( french_lines ) 

length_list = list()
for token_seq in tokenized_french_lines:
    length_list.append( len( token_seq ))
max_output_length = np.array( length_list ).max()
print( 'French max length is {}'.format( max_output_length ))

padded_french_lines = preprocessing.sequence.pad_sequences( tokenized_french_lines , maxlen=max_output_length, padding='post' )
decoder_input_data = np.array( padded_french_lines )
print( 'Decoder input data shape -> {}'.format( decoder_input_data.shape ))

french_word_dict = tokenizer.word_index
num_french_tokens = len( french_word_dict )+1
print( 'Number of French tokens = {}'.format( num_french_tokens))

decoder_target_data = list()
for token_seq in tokenized_french_lines:
    decoder_target_data.append( token_seq[ 1 : ] ) 
    
padded_french_lines = preprocessing.sequence.pad_sequences( decoder_target_data , maxlen=max_output_length, padding='post' )
onehot_french_lines = utils.to_categorical( padded_french_lines , num_french_tokens )
decoder_target_data = np.array( onehot_french_lines )
print( 'Decoder target data shape -> {}'.format( decoder_target_data.shape ))

encoder_inputs = tf.keras.layers.Input(shape=( None , ))
encoder_embedding = tf.keras.layers.Embedding( num_eng_tokens, 256 , mask_zero=True ) (encoder_inputs)
encoder_outputs , state_h , state_c = tf.keras.layers.LSTM( 256 , return_state=True , recurrent_dropout=0.2 , dropout=0.2 )( encoder_embedding )
encoder_states = [ state_h , state_c ]

decoder_inputs = tf.keras.layers.Input(shape=( None ,  ))
decoder_embedding = tf.keras.layers.Embedding( num_french_tokens, 256 , mask_zero=True) (decoder_inputs)
decoder_lstm = tf.keras.layers.LSTM( 256 , return_state=True , return_sequences=True , recurrent_dropout=0.2 , dropout=0.2)
decoder_outputs , _ , _ = decoder_lstm ( decoder_embedding , initial_state=encoder_states )
decoder_dense = tf.keras.layers.Dense( num_french_tokens , activation=tf.keras.activations.softmax ) 
output = decoder_dense ( decoder_outputs )

model = tf.keras.models.Model([encoder_inputs, decoder_inputs], output )
model.compile(optimizer=tf.keras.optimizers.Adam(), loss='categorical_crossentropy')

model.summary()
model.fit([encoder_input_data , decoder_input_data], decoder_target_data, batch_size=250, epochs=25 ) 
model.save( 'model.h5' ) 
def make_inference_models():
    
    encoder_model = tf.keras.models.Model(encoder_inputs, encoder_states)
    
    decoder_state_input_h = tf.keras.layers.Input(shape=( 256,))
    decoder_state_input_c = tf.keras.layers.Input(shape=( 256 ,))
    
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    
    decoder_outputs, state_h, state_c = decoder_lstm(
        decoder_embedding , initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = tf.keras.models.Model(
        [decoder_inputs] + decoder_states_inputs,
        [decoder_outputs] + decoder_states)
    
    return encoder_model , decoder_model
def str_to_tokens( sentence : str ):
    words = sentence.lower().split()
    tokens_list = list()
    for word in words:
        tokens_list.append( eng_word_dict[ word ] ) 
    return preprocessing.sequence.pad_sequences( [tokens_list] , maxlen=max_input_length , padding='post')
enc_model , dec_model = make_inference_models()

enc_model.save( 'enc_model.h5' ) 
dec_model.save( 'dec_model.h5' ) 
model.save( 'model.h5' ) 

for epoch in range( encoder_input_data.shape[0] ):
    states_values = enc_model.predict( str_to_tokens( input( 'Enter eng sentence : ' ) ) )
    #states_values = enc_model.predict( encoder_input_data[ epoch ] )
    empty_target_seq = np.zeros( ( 1 , 1 ) )
    empty_target_seq[0, 0] = french_word_dict['start']
    stop_condition = False
    decoded_translation = ''
    while not stop_condition :
        dec_outputs , h , c = dec_model.predict([ empty_target_seq ] + states_values )
        sampled_word_index = np.argmax( dec_outputs[0, -1, :] )
        sampled_word = None
        for word , index in french_word_dict.items() :
            if sampled_word_index == index :
                decoded_translation += ' {}'.format( word )
                sampled_word = word
        
        if sampled_word == 'end' or len(decoded_translation.split()) > max_output_length:
            stop_condition = True
            
        empty_target_seq = np.zeros( ( 1 , 1 ) )  
        empty_target_seq[ 0 , 0 ] = sampled_word_index
        states_values = [ h , c ] 

    print( decoded_translation )
    