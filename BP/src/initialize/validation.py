from data.data_processing import *
from initialize.generator import load_network
from sklearn.metrics import classification_report
import numpy as np
from itertools import islice


def train_validation(nn_type, x, y):
    model = load_network(nn_type)
    model = fit_model(nn_type, model, x, y)

    model.save('models/' + nn_type + '_val.h5')
    return model


def test(nn_type, x, y, class_names):
    model = load_network(nn_type + '_val')
    if "cnn" in nn_type:
        y_pred = model.predict([x, x, x])
    else:
        y_pred = model.predict(x)
    y_pred_labels = np.argmax(y_pred, axis=1)
    report = classification_report(np.argmax(y, axis=1), y_pred_labels, target_names=class_names)
    print(report)
    file_path = 'reports.txt'
    with open(file_path, 'a') as file:
        file.write(nn_type + " F1-score confusion matrix\n\n" + report + "\n\n\n")
    file.close()


def explain_model(nn_type, x_test, y_test, tokenizer, num_words=10, num_lines=10):
    model = load_network(nn_type + '_val')
    unique_labels = np.unique(y_test, axis=0)  # Get the unique labels from y_test
    selected_entries = []
    labels = []

    for label in unique_labels:
        indices = np.where(y_test == label)[0]  # Get the indices of entries with the current label
        random_indices = np.random.choice(indices, size=num_lines, replace=False)  # Randomly select 10 indices
        selected_entries.extend(x_test[random_indices])  # Append the selected entries to the list
        labels.extend(y_test[random_indices])

    selected_entries = np.array(selected_entries)
    text_rep = tokenizer.sequences_to_texts(selected_entries)
    word_dict = {}
    for i, line in enumerate(selected_entries):
        words = text_rep[i].split()
        emotion = one_hot_to_text(labels[i])
        line = np.reshape(line, (1, 93))
        model.predict(line, verbose=0)
        attention = model.get_layer('attention')
        weights = attention.get_weights()[0]
        words += [''] * (100 - len(words))

        if emotion not in word_dict:
            word_dict[emotion] = {}

        for j, weight in enumerate(weights):
            if words[j] not in word_dict[emotion]:
                word_dict[emotion][words[j]] = [weight]
            else:
                word_dict[emotion][words[j]].append(weight)

    for dictionary in word_dict.values():
        for word, weights in dictionary.items():
            mean_weight = np.mean(weights)  # Compute the mean of the list
            dictionary[word] = mean_weight

    file_path = 'explanations.txt'
    with open(file_path, 'a') as file:
        file.write(nn_type + " most influential words per category, " + str(num_lines) + " samples per emotion\n\n")
        for emotion, dictionary in word_dict.items():
            file.write("Class: " + emotion + ", top " + str(num_words) + " mean weights per word\n\n")
            sorted_dict = dict(sorted(dictionary.items(), key=lambda item: item[1], reverse=True))
            for word, weights in islice(sorted_dict.items(), num_words):
                mean_weight = np.mean(weights)  # Compute the mean of the list
                file.write(word + ": " + str(mean_weight) + "\n")
            file.write("\n\n")
    file.close()


def train_test(nn_type, tokenizer):
    class_names = ['sadness', 'neutral', 'fear', 'disgust', 'joy', 'anger', 'surprise']
    x_train, y_train = load_dataset('datasets/train_data_clean.pkl')
    x_test, y_test = load_dataset('datasets/test_data_clean.pkl')
    train_validation(nn_type, x_train, y_train)
    test(nn_type, x_test, y_test, class_names)
    explain_model(nn_type, x_test, y_test, tokenizer, num_lines=20)


# Selects the right neural network to train depending on the passed string parameter nn_type
def fit_model(nn_type, model, train_x, train_y, val=0):
    if "cnn" in nn_type:
        # Multi-channel CNN, requires as many inputs as there are channels (3)
        model.fit([train_x, train_x, train_x], train_y, epochs=10, verbose=2, validation_split=val)
    elif "rnn" in nn_type:
        model.fit(train_x, train_y, epochs=15, verbose=2, validation_split=val)
    elif "lstm" in nn_type:
        model.fit(train_x, train_y, epochs=60, verbose=2, validation_split=val)
    else:
        if "w2v" in nn_type:
            model.fit(train_x, train_y, epochs=15, verbose=2, validation_split=val)
        else:
            model.fit(train_x, train_y, epochs=8, verbose=2, validation_split=val)
    return model
