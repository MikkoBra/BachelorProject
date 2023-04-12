from keras.models import Model
from keras.layers import Embedding, Dense, Input, SimpleRNN, Attention
from keras.utils import plot_model


def rnn_model(length, voc_size):
    inputs = Input(shape=(length,))
    embedding = Embedding(input_dim=voc_size, output_dim=100)(inputs)
    rnn = SimpleRNN(units=32)(embedding)
    outputs = Dense(units=7, activation='softmax')(rnn)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print(model.summary())
    plot_model(model, show_shapes=True, to_file='rnn.png')
    return model
