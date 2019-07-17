import os
import re
import string
import xml.etree.ElementTree as ET
from pickle import dump
from unicodedata import normalize

from numpy import array

# load doc into memory
def load_doc(filename):
    file = open(filename, mode='rt', encoding='utf-8')
    text = file.read()
    text = text_to_vietkey(text)
    file.close()
    return text

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

def to_pairs(doc):
    lines = doc.strip().split('\n')
    pairs = [line.split('\t') for line in lines]
    for file_name in os.listdir('vn_data_xml'):
        try:
            root = ET.parse('vn_data_xml/' + file_name).getroot()
            for pair in root.findall('text/spair'):
                values = pair.findall('s')
                pair = []
                if '-' in values[1].text:
                    for word in value.text.split(' '):
                        if '-' in word:
                            word.replace('-', ' ')
                for value in values:
                    pair.append(text_to_vietkey(value.text))
                if (len(pair[0].split()) <= 50 and len(pair[1].split()) <= 50):
                    pairs.append(pair)
        except:
            print('error')
    return pairs

# clean a list of lines
def clean_pairs(lines, isBert=False):
    cleaned = list()
    # prepare regex for char filtering
    re_punc = re.compile('[%s]' % re.escape(string.punctuation))
    re_print = re.compile('[^%s]' % re.escape(string.printable)) 
    for pair in lines:
        clean_pair = list()
        for line in pair:
            # normalize unicode characters
            line = normalize('NFD', line).encode('ascii', 'ignore')
            line = line.decode('UTF-8')
            # tokenize on white space
            line = line.split()
            # convert to lowercase
            line = [word.lower() for word in line]
            # remove punctuation from each token
            line = [re_punc.sub('', w) for w in line]
            # remove non-printable chars form each token
            line = [re_print.sub('', w) for w in line]
            # remove tokens with numbers in them
            line = [word for word in line if word.isalpha()]
            # store as string
            clean_pair.append(' '.join(line))
        t = clean_pair[0]
        clean_pair[0] = clean_pair[1]
        clean_pair[1] = t
        if not isBert:
            clean_pair[0] = 'bstart ' + clean_pair[0] + ' kend'
        cleaned.append(clean_pair)
    return array(cleaned)

# save a list of clean sentences to file
def save_clean_data(sentences, filename):
    dump(sentences, open(filename, 'wb'))
    print('Saved: %s' % filename)

# load dataset
filename = 'vie.txt'
doc = load_doc(filename)
# split into english-german pairs
pairs = to_pairs(doc)
# clean sentences
clean_pairs = clean_pairs(pairs)
# save clean pairs to file
save_clean_data(clean_pairs, 'english-german.pkl')
print(clean_pairs.shape)
# spot check
text_file = open("Output.txt", "w")
for i in range(len(clean_pairs)):
    text_file.write('[%s] => [%s]' % (clean_pairs[i,0], clean_pairs[i,1]))
