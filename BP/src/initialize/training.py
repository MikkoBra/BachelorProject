from keras.utils import to_categorical

from data.data_processing import *
from initialize.mlp import mlp_model
from initialize.cnn import cnn_model
from initialize.rnn import rnn_model
from initialize.lstm import lstm_model


def create_model(nn_type, length, voc_size, train_x, train_y):
    if nn_type == "cnn":
        model = cnn_model(length, voc_size)
        model.fit([train_x, train_x, train_x], train_y, epochs=10, verbose=2)
    elif nn_type == "rnn":
        model = rnn_model(length, voc_size)
        model.fit(train_x, train_y, epochs=10, verbose=2)
    elif nn_type == "mlp":
        model = mlp_model(length)
        model.fit(train_x, train_y, epochs=50, verbose=2)
    else:
        model = lstm_model(length, voc_size)
        model.fit(train_x, train_y, epochs=50, verbose=2)
    return model


# Trains a neural network on the cleaned training data
# TODO: attention layer
# TODO: lstm
# TODO: word2vec
def train_network(nn_type):
    # Retrieve train data from file
    train_text, train_labels = load_dataset('train_clean.pkl')
    # Convert labels to one_hot matrices
    one_hot_labels = to_categorical(train_labels)
    # Create tokenizer
    tokenizer = create_tokenizer(train_text)

    length = max_length(train_text)
    voc_size = vocab_size(tokenizer)

    # Convert training string data to padded encoded data
    train_x = encode_and_pad(tokenizer, train_text, length)
    # Create model
    model = create_model(nn_type, length, voc_size, train_x, one_hot_labels)

    # Save model for later evaluation with test set
    model.save('./models/' + nn_type + '.h5')
