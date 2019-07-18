from pickle import dump, load

from numpy import array, expand_dims
from numpy.random import shuffle
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import (LSTM, Dense, Embedding, RepeatVector,
                                     TimeDistributed)
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm

from hdf5DatasetWriter import HDF5DatasetWriter

# load a clean dataset
def load_clean_sentences(filename):
    return load(open(filename, 'rb'))

# save a list of clean sentences to file
def save_clean_data(sentences, filename):
    dump(sentences, open(filename, 'wb'))
    print('Saved: %s' % filename)

# fit a tokenizer
def create_tokenizer(lines):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer

# max sentence length
def max_length(lines):
    return max(len(line.split()) for line in lines)

# encode and pad sequences
def encode_sequences(tokenizer, length, lines, isOutput=False):
    # integer encode sequences
    X = tokenizer.texts_to_sequences(lines)
    if isOutput:
      outX = []
      for x in X:
        outX.append(x[1:])
      outX = pad_sequences(outX, maxlen=length, padding='post')
      return array(outX)
    # pad sequences with 0 values
    X = pad_sequences(X, maxlen=length, padding='post')
    return X

# one hot encode target sequence
def encode_output(sequence, vocab_size):
    encoded = to_categorical(sequence, num_classes=vocab_size)
    return encoded

def write_hdf5(number, file_path, max_len_src, max_len_des, vocab_size_des, X, Y, out):
    print('max_len_src ', out[1])
    hdf5_dataset = HDF5DatasetWriter(input1_dims=(number, max_len_src), input2_dims=(number, max_len_des), label_dims=(number, max_len_des), outputPath=file_path)
    with tqdm(total=number) as pbar:
        for i in range(0, number):
            x = expand_dims(X[i], axis=0)
            y = expand_dims(Y[i], axis=0)
            o = expand_dims(out[i], axis=0)
            hdf5_dataset.add(x, y, o)
            pbar.update(1)
    hdf5_dataset.close()
    return

# load dataset
raw_dataset = load_clean_sentences('data/english-vietnamese-min.pkl')
# reduce dataset size
n_sentences = 3397
dataset = raw_dataset[:n_sentences, :]
# random shuffle
shuffle(dataset)
# split into train/test
train, test = dataset[:int(n_sentences * 0.8)], dataset[int(n_sentences * 0.8):]

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

# prepare training data
trainX = encode_sequences(ger_tokenizer, ger_length, train[:, 1])
trainY = encode_sequences(eng_tokenizer, eng_length, train[:, 0])
outTrainY = encode_sequences(eng_tokenizer, eng_length, train[:, 0], isOutput=True)

# prepare validation data
testX = encode_sequences(ger_tokenizer, ger_length, test[:, 1])
testY = encode_sequences(eng_tokenizer, eng_length, test[:, 0])
outTestY = encode_sequences(eng_tokenizer, eng_length, test[:, 0], isOutput=True)

write_hdf5(int(n_sentences * 0.8), 'hdf5/translate_train_min.hdf5', ger_length, eng_length, eng_vocab_size, trainX, trainY, outTrainY)
write_hdf5(n_sentences - int(n_sentences * 0.8), 'hdf5/translate_test_min.hdf5', ger_length, eng_length, eng_vocab_size, testX, testY, outTestY)
save_clean_data(dataset, 'data/english-vietnamese-both-min.pkl')
save_clean_data(test, 'data/english-vietnamese-test-min.pkl')
