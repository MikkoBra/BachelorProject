from data.data_processing import *
from data.word_intensity import create_weighted_words_image
from initialize.generator import load_network
from sklearn.metrics import classification_report
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

selected_entries = None
labels = []
text_representation = None


def train_validation(nn_type, x, y):
    model = load_network(nn_type)
    model = fit_model(nn_type, model, x, y, 0.25)

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


def load_random_sentences(x_test, y_test, tokenizer):
    global selected_entries, labels, text_representation
    unique_labels = np.unique(y_test, axis=0)  # Get the unique labels from y_test
    selected_entries = []
    labels = []

    for label in unique_labels:
        indices = np.where(np.all(y_test == label, axis=1))[0]
        random_index = np.random.choice(indices, size=1, replace=False)  # Randomly select 1 semtemce
        selected_entries.extend(x_test[random_index])  # Append the selected entry to the list
        labels.extend(y_test[random_index])

    selected_entries = np.array(selected_entries)
    text_representation = tokenizer.sequences_to_texts(selected_entries)


def predict_sentence(model, sentence, idx, nn_type):
    emotion = one_hot_to_text(labels[idx])
    sentence = np.reshape(sentence, (1, 93))
    if "cnn" in nn_type:
        prediction = model.predict([sentence, sentence, sentence], verbose=0)[0]
    else:
        prediction = model.predict(sentence, verbose=0)[0]
    max_val = max(prediction)
    prediction = np.array([1 if x == max_val else 0 for x in prediction])
    predicted_emotion = one_hot_to_text(prediction)
    print("real: " + emotion + ", predicted: " + predicted_emotion)
    return model, emotion, predicted_emotion


def get_weights(model):
    attention_layers = []
    for layer in model.layers:
        if "attention" in layer.name:
            attention_layers.append(model.get_layer(layer.name))
    if len(attention_layers) == 1:
        weights = attention_layers[0].get_weights()[0]
    else:
        weights = np.zeros((100, 100))
        for layer in attention_layers:
            weights += layer.get_weights()[0]
        weights /= len(attention_layers)
    return weights


def update_word_dict(word_dict, sentence, weights, emotion):
    words = sentence.split()
    words += [''] * (100 - len(words))
    if emotion not in word_dict:
        word_dict[emotion] = {}

    for j, weight in enumerate(weights):
        if words[j] not in word_dict[emotion]:
            word_dict[emotion][words[j]] = [weight]
        else:
            word_dict[emotion][words[j]].append(weight)

    return word_dict


def average_word_weights(word_dict):
    for dictionary in word_dict.values():
        for word, weights in dictionary.items():
            mean_weight = np.mean(weights)  # Compute the mean of the list
            dictionary[word] = mean_weight
    return word_dict


def visualize_examples(word_dict, predicted_emotions, nn_type):
    word_list = []
    weight_list = []
    file_path = 'images/example_sentences/'
    for emotion, predicted in zip(word_dict.keys(), predicted_emotions):
        for word, weight in word_dict[emotion].items():
            word_list.append(word)
            weight_list.append(weight)
        title = 'Example sentence for ' + emotion + ', classified as ' + predicted + ' by ' + nn_type
        image = create_weighted_words_image(word_list, weight_list, emotion, title)
        image.save(file_path + nn_type + "/" + emotion + ".png")
        word_list = []
        weight_list = []


def visualize_top_words(word_dict, predicted_emotions, nn_type):
    word_list = []
    weight_list = []
    file_path = 'images/top10words/'
    for emotion, predicted in zip(word_dict.keys(), predicted_emotions):
        word_weights = word_dict[emotion].items()
        sorted_word_weights = sorted(word_weights, key=lambda x: x[1],
                                     reverse=True)
        selected_word_weights = sorted_word_weights[:10]

        for word, weight in selected_word_weights:
            word_list.append(word)
            weight_list.append(weight)

        title = 'Top 10 words for ' + emotion + ', classified as ' + predicted + ' by ' + nn_type
        image = create_weighted_words_image(word_list, weight_list, emotion, title)
        image.save(file_path + nn_type + "/" + emotion + ".png")

        word_list = []
        weight_list = []


def explain_model(nn_type):
    global selected_entries, text_representation
    model = load_network(nn_type + '_val')

    word_dict = {}
    predicted_emotions = []
    for i, sentence in enumerate(selected_entries):
        model, emotion, predicted = predict_sentence(model, sentence, i, nn_type)
        predicted_emotions.append(predicted)
        weights = get_weights(model)
        word_dict = update_word_dict(word_dict, text_representation[i], weights, emotion)

    word_dict = average_word_weights(word_dict)

    visualize_examples(word_dict, predicted_emotions, nn_type)
    visualize_top_words(word_dict, predicted_emotions, nn_type)


def train_test(nn_type):
    class_names = ['sadness', 'neutral', 'fear', 'disgust', 'joy', 'anger', 'surprise']
    x_train, y_train = load_dataset('datasets/train_data_clean.pkl')
    x_test, y_test = load_dataset('datasets/test_data_clean.pkl')
    train_validation(nn_type, x_train, y_train)
    test(nn_type, x_test, y_test, class_names)
    return x_test, y_test


# Selects the right neural network to train depending on the passed string parameter nn_type
def fit_model(nn_type, model, train_x, train_y, val=0.0):
    if "cnn" in nn_type:
        # Multi-channel CNN, requires as many inputs as there are channels (3)
        if "w2v" in nn_type:
            model.fit([train_x, train_x, train_x], train_y, epochs=15, verbose=2, validation_split=val)
        else:
            model.fit([train_x, train_x, train_x], train_y, epochs=15, verbose=2, validation_split=val)
    elif "rnn" in nn_type:
        model.fit(train_x, train_y, epochs=15, verbose=2, validation_split=val)
    elif "lstm" in nn_type:
        model.fit(train_x, train_y, epochs=13, verbose=2, validation_split=val)
    else:
        if "w2v" in nn_type:
            model.fit(train_x, train_y, epochs=15, verbose=2, validation_split=val)
        else:
            model.fit(train_x, train_y, epochs=15, verbose=2, validation_split=val)
    return model
