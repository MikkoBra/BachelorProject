from keras.layers import Embedding, Dense, Input, SimpleRNN
from keras.models import Model
from keras.utils import plot_model

from data.word2vec import embedding_layer


def rnn_model(length, voc_size, tokenizer, w2v=False):
    inputs = Input(shape=(length,))
    if w2v:
        embedding = embedding_layer(tokenizer, voc_size, length, inputs)
    else:
        embedding = Embedding(voc_size, 100)(inputs)
    rnn = SimpleRNN(units=32)(embedding)
    outputs = Dense(units=7, activation='softmax')(rnn)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print(model.summary())
    filename = 'images/rnn_w2v.png' if w2v else 'images/rnn.png'
    plot_model(model, show_shapes=True, to_file=filename)
    return model
