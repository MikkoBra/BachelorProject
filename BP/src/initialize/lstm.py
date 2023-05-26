import tensorflow as tf
from keras.layers import Dense, Dropout, LSTM, Embedding, Input, Reshape
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import plot_model

from data.word2vec import embedding_layer
from initialize.attention import Attention

# fix random seed for reproducibility
tf.random.set_seed(7)


def lstm_model(length, voc_size, tokenizer, w2v):
    # Input layer
    inputs = Input(shape=(length,))
    # Embedding layer (may implement word2vec)
    if w2v:
        embedding = embedding_layer(tokenizer, voc_size, inputs)
    else:
        embedding = Embedding(voc_size, 100)(inputs)
    # LSTM layer
    lstm = LSTM(100)(embedding)
    # Dropout layer
    dropout = Dropout(0.2)(lstm)
    # Reshape to fit attention layer
    reshaped = Reshape((1, 100))(dropout)
    # Attention layer
    attention = Attention()(reshaped)
    # Output layer with 7 classifications
    outputs = Dense(units=7, activation='softmax')(attention)
    # Adjust learning rate to speed up learning
    optimizer = Adam(learning_rate=0.01)
    model = Model(inputs=inputs, outputs=outputs)

    # Compile model
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    filename = 'images/lstm_w2v.png' if w2v else 'images/lstm.png'
    plot_model(model, show_shapes=True, to_file=filename)
    return model
