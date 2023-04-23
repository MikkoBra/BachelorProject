import tensorflow as tf
from keras.layers import Dense, Dropout, LSTM, Embedding, Input
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import plot_model

from data.word2vec import embedding_layer

# fix random seed for reproducibility
tf.random.set_seed(7)


def lstm_model(length, voc_size, tokenizer, w2v):
    inputs = Input(shape=(length,))
    if w2v:
        embedding = embedding_layer(tokenizer, voc_size, length, inputs)
    else:
        embedding = Embedding(voc_size, 100)(inputs)
    lstm = LSTM(100)(embedding)
    dropout = Dropout(0.2)(lstm)
    outputs = Dense(7, activation='softmax')(dropout)
    optimizer = Adam(learning_rate=0.01)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    print(model.summary())
    filename = 'images/lstm_w2v.png' if w2v else 'images/lstm.png'
    plot_model(model, show_shapes=True, to_file=filename)
    return model
