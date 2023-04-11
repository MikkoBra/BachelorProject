from collections import Counter
from pickle import dump
from pickle import load
from string import punctuation

import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.models import load_model
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split


# Reads the training data into a dataframe and reduces it to relevant features
def read_train_data():
    train_data = pd.read_table("../datasets/train_data.tsv", header=0)
    train_data = train_data[['essay', 'emotion']]
    print(train_data.shape)
    return train_data


# Reads the test data into a dataframe and reduces it to relevant features (currently unused, test_data does not contain
# emotion labels)
def read_test_data():
    test_data = pd.read_table("../datasets/test_data.tsv")
    test_data = test_data[['essay']]
    print(test_data.shape)
    return test_data


# Cleans the essays in the dataset
def clean_essay(essay, vocab):
    # Split into tokens
    tokens = essay.split()
    # Remove punctuation
    table = str.maketrans('', '', punctuation)
    tokens = [word.translate(table) for word in tokens]
    # Remove non-alphabetic tokens
    tokens = [word for word in tokens if word.isalpha()]
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if w not in stop_words]
    # Remove single characters
    tokens = [word for word in tokens if len(word) > 1]
    # Convert to lowercase
    tokens = [word.lower() for word in tokens]
    if vocab:
        return tokens
    else:
        # Convert tokens back to string sentence
        token_string = " ".join(tokens)
        return token_string


# Convert string labels to integer labels
def convert_labels(labels):
    label_dict = {'sadness': 0, 'neutral': 1, 'fear': 2, 'disgust': 3, 'joy': 4, 'anger': 5, 'surprise': 6}
    return labels.replace(label_dict)


# Cleans a text dataset and saves the cleaned data to a given filename
def clean_and_save(data, labels, filename):
    lambda_clean = lambda x: clean_essay(x, False)
    # Clean essay texts
    data = data.apply(lambda_clean)
    # Convert labels to integers
    labels = convert_labels(labels)
    dump([data, labels], open(filename, 'wb'))
    print('Saved: %s' % filename)


# Saves the vocabulary of a dataset
def save_vocab(data, filename, min_occurrence):
    vocab = Counter()
    for essay in data:
        tokens = clean_essay(essay, True)
        vocab.update(tokens)
    tokens = [k for k, c in vocab.items() if c >= min_occurrence]
    # convert lines to a single blob of text
    vocab_data = '\n'.join(tokens)
    # open file
    file = open(filename, 'w', encoding="utf-8")
    # write text
    file.write(vocab_data)
    # close file
    file.close()
    print('Saved: %s' % filename)


# Load a dataset file
def load_dataset(filename):
    return load(open(filename, 'rb'))


# Load a text file
def load_txt(filename):
    # open the file as read only
    file = open(filename, 'r')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text


# Creates and fits tokenizer
def create_tokenizer(data):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(data)
    return tokenizer


# Retrieves vocab size of tokenizer
def vocab_size(tokenizer):
    return len(tokenizer.word_index) + 1


# Retrieves max essay length in the dataset
def max_length(data):
    return max([len(essay.split()) for essay in data])


# Encodes text to integers and pads the encoded text to a given max length
def encode_and_pad(tokenizer, data, length):
    encoded = tokenizer.texts_to_sequences(data)
    print(length)
    padded = pad_sequences(encoded, maxlen=length, padding='post')
    return padded


# Cleans text datasets and saves them to local files
def create_clean_data():
    data = read_train_data()
    # data
    x = data['essay']
    # labels
    y = data['emotion']
    # 10/90 split data into train and test (X) data, with labels (Y) for each
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.10, random_state=1)
    clean_and_save(x_train, y_train, 'train_clean.pkl')
    clean_and_save(x_test, y_test, 'test_clean.pkl')
