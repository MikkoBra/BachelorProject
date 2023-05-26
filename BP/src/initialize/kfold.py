from numpy import mean, std
from sklearn.model_selection import KFold

from data.data_processing import load_dataset, preprocess_clean_data, oversample
from initialize.generator import load_network

from sklearn.metrics import classification_report, accuracy_score
import numpy as np


def kfold(nn_type):
    x, y = load_dataset('datasets/all_data_clean.pkl')
    data, one_hot = preprocess_clean_data(x, y)
    data, one_hot = oversample(data, one_hot)
    # prepare the cross-validation procedure
    kf = KFold(n_splits=5, shuffle=True, random_state=1)
    scores = []

    i = 1
    for train_index, test_index in kf.split(data):
        x_train, y_train = data[train_index], one_hot[train_index]
        x_test, y_test = data[test_index], one_hot[test_index]

        model = load_network(nn_type)
        fit_model(nn_type, model, x_train, y_train)

        if "cnn" in nn_type:
            y_pred = model.predict([x_test, x_test, x_test])
        else:
            y_pred = model.predict(x_test)
        y_pred_labels = np.argmax(y_pred, axis=1)
        class_names = ['sadness', 'neutral', 'fear', 'disgust', 'joy', 'anger', 'surprise']
        report = classification_report(np.argmax(y_test, axis=1), y_pred_labels, target_names=class_names)
        print(report)
        file_path = 'reports.txt'
        with open(file_path, 'a') as file:
            file.write(nn_type + " F1-score confusion matrix, split " + str(i) + "\n\n" + report + "\n\n\n")
        file.close()
        i += 1
        accuracy = accuracy_score(np.argmax(y_test, axis=1), y_pred_labels)
        scores.append(accuracy)

    print('Average accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))


# Selects the right neural network to train depending on the passed string parameter nn_type
def fit_model(nn_type, model, train_x, train_y, val=0):
    if "cnn" in nn_type:
        # Multi-channel CNN, requires as many inputs as there are channels (3)
        model.fit([train_x, train_x, train_x], train_y, epochs=6, verbose=2, validation_split=val)
    elif "rnn" in nn_type:
        model.fit(train_x, train_y, epochs=15, verbose=2, validation_split=val)
    elif "lstm" in nn_type:
        model.fit(train_x, train_y, epochs=60, verbose=2, validation_split=val)
    else:
        model.fit(train_x, train_y, epochs=50, verbose=2, validation_split=val)
    return model
