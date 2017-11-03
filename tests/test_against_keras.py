"""Compare an update against an identical network in Keras."""

import keras
import numpy as np
from keras.layers import Dense, Input
from keras.models import Model
from keras.optimizers import SGD

import simplenet as sn

data = [[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1], [1, 1, 0]]
targets = [[0, 1], [1, 0], [1, 0], [0, 1], [1, 0]]

learning_rate = 0.1
dtype = 'float64'

nn = sn.SimpleNet(
    input_shape=(None, len(data[0])), output_shape=(None, len(targets[0])),
    hidden_layer_sizes=[9, 7, 5, 3],
    learning_rate=learning_rate,
    activation_function=sn.sigmoid,
    output_activation=sn.softmax,
    loss_function=sn.cross_entropy,
    dtype=dtype,
    )

keras.backend.set_floatx(dtype)
print('keras epsilon: {}'.format(keras.backend.epsilon()))

inputs = Input(shape=(len(data[0]),), dtype=dtype)
x = Dense(9, activation='sigmoid')(inputs)
x = Dense(7, activation='sigmoid')(x)
x = Dense(5, activation='sigmoid')(x)
x = Dense(3, activation='sigmoid')(x)
outputs = Dense(len(targets[0]), activation='softmax')(x)

model = Model(inputs=inputs, outputs=outputs)
optimizer = SGD(lr=learning_rate)
model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        )

model.set_weights([layer for weight, bias in zip(nn.weights, nn.biases)
                   for layer in (weight, bias.reshape(-1))])

nn.learn(inputs=data, targets=targets)
history = model.fit(data, targets, epochs=1, verbose=False)


def test_weights_and_biases() -> None:
    """Ensure SimpleNet weights and biases match keras after 1 epoch."""
    # `model.get_weights()` returns a list of alternating weights and biases
    pairs = {
        'weights': (model.get_weights()[::2], nn.weights),
        'biases': (model.get_weights()[1::2], nn.biases),
    }

    for k, pair in pairs.items():
        for idx, (keras_data, nn_data) in enumerate(zip(*pair)):
            assert np.allclose(keras_data, nn_data)


def test_err() -> None:
    """Ensure SimpleNet error matches keras after 1 epoch."""
    assert np.isclose(nn.err, history.history['loss'][0])
