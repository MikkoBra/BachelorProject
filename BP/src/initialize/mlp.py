from keras.layers import Dense, Input, Attention
from keras.models import Model
from keras.utils import plot_model


# Define a simple feed forward (multi-layer perceptron) neural network model
def mlp_model(length):
    # Input layer
    inputs = Input(shape=(length,))
    # Dense layers
    dense1 = Dense(64, activation='relu')(inputs)
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
    plot_model(model, show_shapes=True, to_file='mlp.png')
    return model
