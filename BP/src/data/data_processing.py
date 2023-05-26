from collections import Counter
from pickle import dump
from pickle import load
from string import punctuation
from imblearn.over_sampling import RandomOverSampler
from data.graphs import class_percent_graph
import os
import numpy as np

import nltk
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.utils import to_categorical
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


def one_hot_to_text(one_hot_label):
    label_dict = {0: 'sadness', 1: 'neutral', 2: 'fear', 3: 'disgust', 4: 'joy', 5: 'anger', 6: 'surprise'}
    return label_dict.get(np.where(one_hot_label == 1)[0][0])


# Cleans a text dataset and saves the cleaned data to a given filename
def clean(data, labels):
    lambda_clean = lambda x: clean_essay(x, False)
    # Clean essay texts
    data = data.apply(lambda_clean)
    # Convert labels to integers
    labels = convert_labels(labels)
    return data, labels


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
    padded = pad_sequences(encoded, maxlen=length, padding='post')
    return padded


# Returns cleaned data, and one_hot encoded labels
def preprocess_clean_data(x, y):
    tokenizer = create_tokenizer(x)
    length = max_length(x)
    data = encode_and_pad(tokenizer, x, length)
    one_hot = to_categorical(y)
    return data, one_hot


# Cleans text datasets and saves them to local files
def create_clean_data():
    data = read_train_data()
    # data
    x = data['essay']
    # labels
    y = data['emotion']
    clean_data, clean_labels = clean(x, y)
    dump([clean_data, clean_labels], open('datasets/all_data_clean.pkl', 'wb'))
    print('Saved: datasets/all_data_clean.pkl')


def oversample(x_train, y_train):
    # Apply oversampling to the training data
    oversampler = RandomOverSampler(sampling_strategy='not majority', random_state=1)
    x_train_resampled, y_train_resampled = oversampler.fit_resample(x_train, y_train)
    return x_train_resampled, y_train_resampled


# 80/20 split data into train and test (x) data, with labels (y) for each
def split_data(x, y, test_size):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=1)
    x_train, y_train = oversample(x_train, y_train)
    return x_train, x_test, y_train, y_test


# Split data into train/test, and save into new .pkl files
def save_train_test():
    labels = ['sadness', 'neutral', 'fear', 'disgust', 'joy', 'anger', 'surprise']
    x, y = load_dataset('datasets/all_data_clean.pkl')
    clean_x, clean_y = preprocess_clean_data(x, y)
    class_percent_graph(labels)

    x_train, x_test, y_train, y_test = split_data(clean_x, clean_y, 0.20)

    dump([x_train, y_train], open('datasets/train_data_clean.pkl', 'wb'))
    print('Saved: datasets/train_data_clean.pkl')
    dump([x_train, y_train], open('datasets/test_data_clean.pkl', 'wb'))
    print('Saved: datasets/test_data_clean.pkl')
    class_percent_graph(labels, oversampled=True)


# Turn a list of (sentence) strings into a 2D list of tokens
def sentences_to_lists(data):
    tokenized = []
    for sentence in data:
        tokens = nltk.word_tokenize(sentence)
        tokenized.append(tokens)
    return tokenized


def remove_file(file_path):
    try:
        os.remove(file_path)
        print(f"File '{file_path}' deleted successfully.")
    except OSError as e:
        print(f"Error deleting the file '{file_path}': {e}")


def one_hot_to_int(labels):
    int_labels = np.argmax(labels, axis=1)
    int_labels = int_labels.tolist()
    return int_labels