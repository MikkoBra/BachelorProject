from gensim.models import Word2Vec
from keras.layers import Embedding
from numpy import asarray, zeros

from data.data_processing import load_dataset
from data.data_processing import sentences_to_lists


# Generates word2vec embedding and saves it to a .txt file
def word2vec():
    data, labels = load_dataset('datasets/all_data_clean.pkl')
    sentences = sentences_to_lists(data)
    model = Word2Vec(sentences, workers=8, min_count=1)
    print('Vocabulary size: %d' % len(model.wv))
    filename = 'models/embedding_word2vec.txt'
    model.wv.save_word2vec_format(filename, binary=False)


def load_embedding(filename):
    # load embedding into memory, skip first line
    file = open(filename, 'r')
    lines = file.readlines()[1:]
    file.close()
    # create a map of words to vectors
    embedding = dict()
    for line in lines:
        parts = line.split()
        # key is string word, value is numpy array for vector
        embedding[parts[0]] = asarray(parts[1:], dtype='float32')
    return embedding


# create a weight matrix for the Embedding layer from a loaded embedding
def get_weight_matrix(embedding, vocab):
    # total vocabulary size plus 0 for unknown words
    vocab_size = len(vocab) + 1
    # define weight matrix dimensions with all 0
    weight_matrix = zeros((vocab_size, 100))
    # step vocab, store vectors using the Tokenizer's integer mapping
    for word, i in vocab.items():
        weight_matrix[i] = embedding.get(word)
    return weight_matrix


def embedding_layer(tokenizer, vocab_size, length, inputs):
    # load embedding from file
    raw_embedding = load_embedding('models/embedding_word2vec.txt')
    # get vectors in the right order
    embedding_vectors = get_weight_matrix(raw_embedding, tokenizer.word_index)
    # create the embedding layer
    layer = Embedding(vocab_size, 100, weights=[embedding_vectors], input_length=length, trainable=False)(inputs)
    return layer
