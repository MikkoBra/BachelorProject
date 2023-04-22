from initialize.generator import save_network
from data.data_processing import create_clean_data
from initialize.validation import train_test
from initialize.kfold import kfold
from data.data_processing import save_train_test

nn_types = ['cnn', 'rnn', 'mlp', 'lstm']

if __name__ == '__main__':
    # Clean the tokens in the text data
    # create_clean_data()
    #
    # # Create models of each network and save them to files
    # for nn_type in nn_types:
    #     save_network(nn_type)
    #
    # # Evaluate the models
    # for nn_type in nn_types:
    #     train_test(nn_type)
    train_test('mlp')
