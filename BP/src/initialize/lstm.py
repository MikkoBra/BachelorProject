import tensorflow as tf
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, LSTM, Embedding, Input, Attention
from keras.utils import plot_model
from keras.optimizers import Adam
# fix random seed for reproducibility
tf.random.set_seed(7)


def lstm_model(length, voc_size):
    inputs = Input(shape=(length,))
    embedding = Embedding(input_dim=voc_size, output_dim=100)(inputs)
    lstm = LSTM(100)(embedding)
    dropout = Dropout(0.2)(lstm)
    outputs = Dense(7, activation='softmax')(dropout)
    optimizer = Adam(learning_rate=0.01)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    print(model.summary())
    plot_model(model, show_shapes=True, to_file='images/lstm.png')
    return model
