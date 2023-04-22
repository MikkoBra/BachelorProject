from data.data_processing import *
from initialize.generator import load_network


def fit_validation(nn_type, model, train_x, train_y):
    if nn_type == "cnn":
        model.fit([train_x, train_x, train_x], train_y, epochs=10, verbose=2, validation_split=0.25)
    elif nn_type == "rnn":
        model.fit(train_x, train_y, epochs=10, verbose=2, validation_split=0.25)
    else:
        model.fit(train_x, train_y, epochs=50, verbose=2, validation_split=0.25)
    return model


def train_validation(nn_type):
    x, y = load_dataset('train_data_clean.pkl')
    model = load_network(nn_type)
    model = fit_validation(nn_type, model, x, y)

    model.save('models/' + nn_type + '_val.h5')
    return model


def test(nn_type):
    x, y = load_dataset('test_data_clean.pkl')
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
