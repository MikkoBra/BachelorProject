from keras.layers import Dense, Reshape
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers import Flatten
from keras.layers import Input
from keras.layers import concatenate
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.models import Model
from keras.utils import plot_model
import tensorflow as tf

from data.word2vec import embedding_layer
from initialize.attention import Attention

# fix random seed for reproducibility
tf.random.set_seed(7)


# Define a deep convolutional neural network model
def cnn_model(length, voc_size, tokenizer, w2v=False):
    # Create channels of multi-channel CNN
    flat1, inputs1 = channel(length, voc_size, tokenizer, w2v, 2)
    flat2, inputs2 = channel(length, voc_size, tokenizer, w2v, 4)
    flat3, inputs3 = channel(length, voc_size, tokenizer, w2v, 6)
    # merge
    merged = concatenate([flat1, flat2, flat3])
    # interpretation
    dense1 = Dense(10, activation='relu')(merged)
    # Reshape to fit attention layer
    reshaped = Reshape((1, 10))(dense1)
    attention = Attention()(reshaped)
    # Output layer
    outputs = Dense(units=7, activation='softmax')(attention)
    model = Model(inputs=[inputs1, inputs2, inputs3], outputs=outputs)

    # compile
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # summarize
    print(model.summary())
    filename = 'images/cnn_w2v.png' if w2v else 'images/cnn.png'
    plot_model(model, show_shapes=True, to_file=filename)
    return model


# Creates a single channel in the multi-channel CNN
def channel(length, voc_size, tokenizer, w2v, k_size):
    # Input layer
    inputs = Input(shape=(length,))
    # Embedding layer (may implement word2vec)
    if w2v:
        embedding = embedding_layer(tokenizer, voc_size, length, inputs)
    else:
        embedding = Embedding(voc_size, 100)(inputs)
    # Convolutional layer
    conv = Conv1D(filters=32, kernel_size=k_size, activation='relu')(embedding)
    # Dropout layer
    drop = Dropout(0.5)(conv)
    # Pooling layer
    pool = MaxPooling1D(pool_size=2)(drop)
    # Flattening layer
    flat = Flatten()(pool)
    return flat, inputs
