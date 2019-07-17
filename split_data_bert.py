from pickle import dump, load

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from keras.callbacks import ModelCheckpoint
from keras.layers import LSTM, Dense, Embedding, RepeatVector, TimeDistributed
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from numpy import array, expand_dims
from numpy.random import shuffle
from tqdm import tqdm

from bert.tokenization import FullTokenizer
from hdf5DatasetWriter import HDF5DatasetWriterBert

sess = tf.Session()

def write_hdf5(number, file_path, max_len_src, max_len_des, inputs, masks, segments, outputs):
    hdf5_dataset = HDF5DatasetWriterBert(input1_dims=(number, max_len_src), input2_dims=(number, max_len_des), input3_dims=(number, max_len_des), label_dims=(number, max_len_des), outputPath=file_path)
    with tqdm(total=number) as pbar:
        for i in range(0, number):
            input = expand_dims(inputs[i], axis=0)
            mask = expand_dims(masks[i], axis=0)
            segment = expand_dims(segments[i], axis=0)
            output = expand_dims(outputs[i], axis=0)
            hdf5_dataset.add(input, mask, segment, output)
            pbar.update(1)
    hdf5_dataset.close()
    return

# max sentence length
def max_length(lines):
    return max(len(line.split()) for line in lines)

# load a clean dataset
def load_clean_sentences(filename):
    return load(open(filename, 'rb'))

def create_tokenizer_from_hub_module(bert_path):
    """Get the vocab file and casing info from the Hub module."""
    bert_module = hub.Module(bert_path)
    tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
    vocab_file, do_lower_case = sess.run(
        [tokenization_info["vocab_file"], tokenization_info["do_lower_case"]]
    )
    return FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)

raw_dataset = load_clean_sentences('english-german-bert.pkl')
n_sentences = 190000
dataset = raw_dataset[:n_sentences, :]
shuffle(dataset)

bert_path = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"
tokenizer = create_tokenizer_from_hub_module(bert_path)
encode_length = max_length(dataset[:, 0])
input_ids = []
input_masks = []
segment_ids = []
for encode in dataset[:, 0]:
    # pharse 1
    tokens_encode = tokenizer.tokenize(encode)
    tokens = []
    segment_id = []
    tokens.append("[CLS]")
    segment_id.append(0)
    for token in tokens_encode:
        tokens.append(token)
        segment_id.append(0)
    tokens.append("[SEP]")
    segment_id.append(0)
    input_id = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_id)
    # pharse 2
    input_ids.append(input_id)
    input_masks.append(input_mask)
    segment_ids.append(segment_id)

input_ids = pad_sequences(input_ids, maxlen=encode_length, padding='post')
input_masks = pad_sequences(input_masks, maxlen=encode_length, padding='post')
segment_ids = pad_sequences(segment_ids, maxlen=encode_length, padding='post')

decode_length = max_length(dataset[:, 1])
vocal_size = 0
output_ids = []
for decode in dataset[:, 1]:
    # pharse 1
    tokens_decode = tokenizer.tokenize(decode)
    tokens = []
    for token in tokens_decode:
        tokens.append(token)
    tokens.append("[SEP]")
    output_id = tokenizer.convert_tokens_to_ids(tokens)
    # pharse 2
    local_max = np.max(output_id)
    if local_max >= vocal_size:
        vocal_size = local_max
    # pharse 3
    output_ids.append(output_id)

output_ids = pad_sequences(output_ids, maxlen=decode_length, padding='post')
write_hdf5(171000, 'bert_train.hdf5', encode_length, decode_length, input_ids[:171000], input_masks[:171000], segment_ids[:171000], output_ids[:171000])
write_hdf5(19000, 'bert_test.hdf5', encode_length, decode_length, input_ids[171000:], input_masks[171000:], segment_ids[171000:], output_ids[171000:])