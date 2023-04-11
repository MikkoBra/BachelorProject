from initialize.training import train_network
from data.data_processing import create_clean_data

if __name__ == '__main__':
    # create_clean_data()
    train_network('cnn')
    train_network('rnn')
    # train_network('mlp')
    # train_network('lstm')
