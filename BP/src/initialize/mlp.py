from keras.layers import Dense, Input, Flatten, Reshape
from keras.models import Model
from keras.utils import plot_model
import tensorflow as tf

from data.word2vec import embedding_layer
from initialize.attention import Attention

# fix random seed for reproducibility
tf.random.set_seed(7)


# Define a simple feed forward (multi-layer perceptron) neural network model
def mlp_model(length, voc_size, tokenizer, w2v=False):
    # Input layer
    inputs = Input(shape=(length,))
    # Embedding layer (may implement word2vec)
    if w2v:
        embedding = embedding_layer(tokenizer, voc_size, length, inputs)
        flat = Flatten()(embedding)
        # Dense layers
        dense0 = Dense(128, activation='relu')(flat)
        dense1 = Dense(64, activation='relu')(dense0)
    else:
        dense1 = Dense(64, activation='relu')(inputs)
    # Rest of the layers
    model = model_output(dense1, inputs)

    filename = 'images/mlp_w2v.png' if w2v else 'images/mlp.png'
    plot_model(model, show_shapes=True, to_file=filename)
    return model


def model_output(prev_layer, inputs):
    # Dense layers
    dense2 = Dense(32, activation='relu')(prev_layer)
    dense3 = Dense(16, activation='relu')(dense2)
    # Reshape to fit attention layer
    reshaped = Reshape((1, 16))(dense3)
    attention = Attention()(reshaped)
    # Output layer
    outputs = Dense(units=7, activation='softmax')(attention)
    model = Model(inputs=inputs, outputs=outputs)

    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # summarize
    print(model.summary())
    return model
