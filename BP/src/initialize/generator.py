from keras.models import load_model

from data.data_processing import load_dataset, create_tokenizer, max_length, vocab_size
from initialize.attention import Attention
from initialize.cnn import cnn_model
from initialize.lstm import lstm_model
from initialize.mlp import mlp_model
from initialize.rnn import rnn_model


# Selects the right neural network to generate depending on the passed string parameter nn_type
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
    # Create model with word2vec
    model = create_model(nn_type, length, voc_size, tokenizer, w2v=True)
    model.save('models/' + nn_type + '_w2v.h5')


# Loads a network from a local .h5 file
def load_network(filename):
    custom_objects = {"Attention": Attention}
    return load_model('models/' + filename + '.h5', custom_objects=custom_objects)


# Selects the right neural network to train depending on the passed string parameter nn_type
def fit_model(nn_type, model, train_x, train_y, val=0):
    if nn_type == "cnn" or nn_type == "cnn_w2v":
        # Multi-channel CNN, requires as many inputs as there are channels (3)
        model.fit([train_x, train_x, train_x], train_y, epochs=10, verbose=2, validation_split=val)
    elif nn_type == "rnn" or nn_type == "rnn_w2v":
        model.fit(train_x, train_y, epochs=10, verbose=2, validation_split=val)
    else:
        model.fit(train_x, train_y, epochs=50, verbose=2, validation_split=val)
    return model
