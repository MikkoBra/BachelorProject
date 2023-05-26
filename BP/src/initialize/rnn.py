from keras.layers import Embedding, Dense, Input, SimpleRNN
from keras.models import Model
from keras.utils import plot_model
import tensorflow as tf

from data.word2vec import embedding_layer
from initialize.attention import Attention

# fix random seed for reproducibility
tf.random.set_seed(7)


def rnn_model(length, voc_size, tokenizer, w2v=False):
    # Input layer
    inputs = Input(shape=length)
    # Embedding if using word2vec
    if w2v:
        embedding = embedding_layer(tokenizer, voc_size, inputs)
    else:
        embedding = Embedding(voc_size, 100)(inputs)
    # RNN layer
    rnn = SimpleRNN(units=32, return_sequences=True)(embedding)
    # Attention layer
    attention_layer = Attention()(rnn)
    # Output layer
    outputs = Dense(units=7, activation='softmax', trainable=True)(attention_layer)
    model = Model(inputs, outputs)

    # Compile
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    filename = 'images/rnn_w2v.png' if w2v else 'images/rnn.png'
    plot_model(model, show_shapes=True, to_file=filename)
    return model
