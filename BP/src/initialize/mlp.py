from keras.layers import Embedding, Dense, Input, Flatten, Reshape
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
    inputs = Input(shape=(length,), name='input')
    # Embedding layer (may implement word2vec)
    if w2v:
        embedding = embedding_layer(tokenizer, voc_size, inputs)
    else:
        embedding = Embedding(voc_size, 100)(inputs)
    attention = Attention(name="attention")(embedding)
    # Rest of the layers
    model = model_output(attention, inputs)

    filename = 'images/model architecture/mlp_w2v.png' if w2v else 'images/model architecture/mlp.png'
    plot_model(model, show_shapes=True, to_file=filename)
    return model


def model_output(prev_layer, inputs):
    # Dense layers
    dense0 = Dense(128, activation='relu')(prev_layer)
    dense1 = Dense(64, activation='relu')(dense0)
    dense2 = Dense(32, activation='relu')(dense1)
    dense3 = Dense(16, activation='relu')(dense2)
    # Output layer
    outputs = Dense(units=7, activation='softmax')(dense3)
    model = Model(inputs=inputs, outputs=outputs)

    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
