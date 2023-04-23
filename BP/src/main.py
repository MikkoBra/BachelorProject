from data.data_processing import create_clean_data, save_train_test
from data.word2vec import word2vec
from initialize.generator import save_network
from initialize.validation import train_test

nn_types = ['cnn', 'rnn', 'mlp', 'lstm']

if __name__ == '__main__':
    # Clean the tokens in the text data
    create_clean_data()
    save_train_test()
    word2vec()

    # Create models of each network and save them to files
    # for nn_type in nn_types:
    #     save_network(nn_type)
    save_network('rnn')
    train_test('rnn')
    train_test('rnn_w2v')
