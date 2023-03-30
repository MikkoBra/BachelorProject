from models import nn_classifier

if __name__ == '__main__':
    nn_classifier.create_clean_data()
    nn_classifier.train_network()
