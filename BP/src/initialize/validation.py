from data.data_processing import *
from initialize.generator import load_network, fit_model


def train_validation(nn_type):
    x, y = load_dataset('datasets/train_data_clean.pkl')
    model = load_network(nn_type)
    model = fit_model(nn_type, model, x, y, 0.25)

    model.save('models/' + nn_type + '_val.h5')
    return model


def test(nn_type):
    x, y = load_dataset('datasets/test_data_clean.pkl')
    model = load_network(nn_type + '_val')
    if nn_type == 'cnn':
        scores = model.evaluate([x, x, x], y)
    else:
        scores = model.evaluate(x, y)
    print("Test Loss:", scores[0])
    print("Test Accuracy:", scores[1])


def train_test(nn_type):
    train_validation(nn_type)
    test(nn_type)
