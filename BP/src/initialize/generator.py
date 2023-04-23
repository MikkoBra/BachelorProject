from keras.models import load_model

from data.data_processing import load_dataset, create_tokenizer, max_length, vocab_size
from initialize.cnn import cnn_model
from initialize.lstm import lstm_model
from initialize.mlp import mlp_model
from initialize.rnn import rnn_model


def create_model(nn_type, length, voc_size, tokenizer, w2v=False):
    if nn_type == "cnn":
        model = cnn_model(length, voc_size, tokenizer, w2v)
    elif nn_type == "rnn":
        model = rnn_model(length, voc_size, tokenizer, w2v)
    elif nn_type == "mlp":
        model = mlp_model(length, voc_size, tokenizer, w2v)
    else:
        model = lstm_model(length, voc_size, tokenizer, w2v)
    return model


# Trains a neural network on the cleaned training data
# TODO: attention layer
# TODO: word2vec
def save_network(nn_type):
    # Retrieve train data from file
    x, y = load_dataset('datasets/all_data_clean.pkl')
    # Create tokenizer
    tokenizer = create_tokenizer(x)

    length = max_length(x)
    voc_size = vocab_size(tokenizer)

    # Create model
    model = create_model(nn_type, length, voc_size, tokenizer)
    model.save('models/' + nn_type + '.h5')
    model = create_model(nn_type, length, voc_size, tokenizer, w2v=True)
    model.save('models/' + nn_type + '_w2v.h5')


def load_network(nn_type):
    return load_model('models/' + nn_type + '.h5')


def fit_model(nn_type, model, train_x, train_y, val=0):
    if nn_type == "cnn":
        model.fit([train_x, train_x, train_x], train_y, epochs=10, verbose=2, validation_split=val)
    elif nn_type == "rnn":
        model.fit(train_x, train_y, epochs=10, verbose=2, validation_split=val)
    else:
        model.fit(train_x, train_y, epochs=50, verbose=2, validation_split=val)
    return model
