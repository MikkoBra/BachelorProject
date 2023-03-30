from sklearn.model_selection import train_test_split
from numpy import array
from keras.utils import plot_model
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers import concatenate
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from data.data_processing import *


def create_clean_data():
    data = read_train_data()
    # data
    x = data['essay']
    # labels
    y = data['emotion']
    # 10/90 split data into train and test (X) data, with labels (Y) for each
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.10, random_state=1)
    clean_and_save(x_train, y_train, 'train_clean.pkl')
    clean_and_save(x_test, y_test, 'test_clean.pkl')


# define the model
def define_model(length, voc_size):
    # channel 1
    inputs1 = Input(shape=(length,))
    embedding1 = Embedding(voc_size, 100)(inputs1)
    conv1 = Conv1D(filters=32, kernel_size=4, activation='relu')(embedding1)
    drop1 = Dropout(0.5)(conv1)
    pool1 = MaxPooling1D(pool_size=2)(drop1)
    flat1 = Flatten()(pool1)
    # channel 2
    inputs2 = Input(shape=(length,))
    embedding2 = Embedding(voc_size, 100)(inputs2)
    conv2 = Conv1D(filters=32, kernel_size=6, activation='relu')(embedding2)
    drop2 = Dropout(0.5)(conv2)
    pool2 = MaxPooling1D(pool_size=2)(drop2)
    flat2 = Flatten()(pool2)
    # channel 3
    inputs3 = Input(shape=(length,))
    embedding3 = Embedding(voc_size, 100)(inputs3)
    conv3 = Conv1D(filters=32, kernel_size=8, activation='relu')(embedding3)
    drop3 = Dropout(0.5)(conv3)
    pool3 = MaxPooling1D(pool_size=2)(drop3)
    flat3 = Flatten()(pool3)
    # merge
    merged = concatenate([flat1, flat2, flat3])
    # interpretation
    dense1 = Dense(10, activation='relu')(merged)
    outputs = Dense(7, activation='softmax')(dense1)
    model = Model(inputs=[inputs1, inputs2, inputs3], outputs=outputs)
    # compile
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # summarize
    print(model.summary())
    plot_model(model, show_shapes=True, to_file='multichannel.png')
    return model


def train_network():
    train_text, train_labels = load_dataset('train_clean.pkl')
    one_hot_labels = to_categorical(train_labels)
    tokenizer = create_tokenizer(train_text)
    length = max_length(train_text)
    voc_size = vocab_size(tokenizer)
    print(length, voc_size)
    train_x = pad_tokenizer(tokenizer, train_text, length)
    model = define_model(length, voc_size)
    print(train_x.shape)
    model.fit([train_x, train_x, train_x], one_hot_labels, epochs=20, verbose=2)
    model.save('nn_model.h5')

