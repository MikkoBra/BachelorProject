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
    # Reshape input to match LSTM layer input shape
    reshaped_input = Reshape((length, 1))(inputs)
    # Embedding layer (may implement word2vec)
    if w2v:
        embedding = embedding_layer(tokenizer, voc_size, reshaped_input)
    else:
        embedding = Embedding(voc_size, 100)(reshaped_input)
    # Attention layer
    attention = Attention(name="attention")(embedding)
    # LSTM layer
    lstm = LSTM(32)(attention)
    # Dropout layer
    dropout = Dropout(0.2)(lstm)
    # Rest of the layers
    model = model_output(dropout, inputs)

    filename = 'images/model architecture/lstm_w2v.png' if w2v else 'images/model architecture/lstm.png'
    plot_model(model, show_shapes=True, to_file=filename)
    return model


def model_output(prev_layer, inputs):
    # Dense layers
    dense2 = Dense(32, activation='relu')(prev_layer)
    dense3 = Dense(16, activation='relu')(dense2)
    # Output layer
    outputs = Dense(units=7, activation='softmax')(dense3)
    model = Model(inputs=inputs, outputs=outputs)

    # Compile model
    optimizer = Adam(learning_rate=0.01)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model