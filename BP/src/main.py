from data.data_processing import create_clean_data, save_train_test
from data.word2vec import word2vec
from initialize.generator import save_network
from initialize.validation import train_test

nn_types = ['cnn', 'rnn', 'mlp', 'lstm']

if __name__ == '__main__':
    # Clean the tokens in the text data, save cleaned dataset
    create_clean_data()
    # Split cleaned data into train/test
    save_train_test()
    # Generate word2vec embedding file
    word2vec()

    # Create models of each network and save them to files, and test the networks using a 60/20/20 train/test/validation
    # split
    for nn_type in nn_types:
        save_network(nn_type)
        train_test(nn_type)
