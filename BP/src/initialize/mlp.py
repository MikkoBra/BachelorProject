from keras.layers import Dense, Input, Flatten
from keras.models import Model
from keras.utils import plot_model

from data.word2vec import embedding_layer


# Define a simple feed forward (multi-layer perceptron) neural network model
def mlp_model(length, voc_size, tokenizer, w2v=False):
    # Input layer
    inputs = Input(shape=(length,))
    if w2v:
        embedding = embedding_layer(tokenizer, voc_size, length, inputs)
        flat = Flatten()(embedding)
        # Dense layers
        dense0 = Dense(128, activation='relu')(flat)
        dense1 = Dense(64, activation='relu')(dense0)
    else:
        dense1 = Dense(64, activation='relu')(inputs)
    model = model_output(dense1, inputs)
    filename = 'images/mlp_w2v.png' if w2v else 'images/mlp.png'
    plot_model(model, show_shapes=True, to_file=filename)
    return model


def model_output(dense1, inputs):
    dense2 = Dense(32, activation='relu')(dense1)
    dense3 = Dense(16, activation='relu')(dense2)
    # Attention layer
    # implementation of attention layer here
    # Output layer
    outputs = Dense(7, activation='softmax')(dense3)
    model = Model(inputs=inputs, outputs=outputs)
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # summarize
    print(model.summary())
    return model
