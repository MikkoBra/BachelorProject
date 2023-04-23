from numpy import mean, std
from sklearn.model_selection import KFold

from data.data_processing import load_dataset, preprocess_clean_data
from initialize.generator import load_network, fit_model


def kfold(nn_type):
    x, y = load_dataset('datasets/all_data_clean.pkl')
    data, one_hot = preprocess_clean_data(x, y)
    # prepare the cross-validation procedure
    kf = KFold(n_splits=5, shuffle=True, random_state=1)
    scores = []

    for train_index, test_index in kf.split(data):
        x_train, y_train = data[train_index], one_hot[train_index]
        x_test, y_test = data[test_index], one_hot[test_index]

        model = load_network(nn_type)
        fit_model(nn_type, model, x_train, y_train)

        if nn_type == 'cnn':
            score = model.evaluate([x_test, x_test, x_test], y_test)
        else:
            score = model.evaluate(x_test, y_test)
        scores.append(score[1])

    print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
