from data.data_processing import load_dataset, create_tokenizer, max_length, vocab_size
from initialize.mlp import mlp_model
from initialize.cnn import cnn_model
from initialize.rnn import rnn_model
from initialize.lstm import lstm_model
from keras.models import load_model


def create_model(nn_type, length, voc_size):
    if nn_type == "cnn":
        model = cnn_model(length, voc_size)
    elif nn_type == "rnn":
        model = rnn_model(length, voc_size)
    elif nn_type == "mlp":
        model = mlp_model(length)
    else:
        model = lstm_model(length, voc_size)
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
    model = create_model(nn_type, length, voc_size)
    model.save('models/' + nn_type + '.h5')


def load_network(nn_type):
    return load_model('models/' + nn_type + '.h5')
