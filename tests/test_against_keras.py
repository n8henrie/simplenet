"""Compare an update against an identical network in Keras."""

from copy import deepcopy

import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD

import simplenet as sn

dtype = "float64"
keras.backend.set_floatx(dtype)

inputs = [[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1], [1, 1, 0]]
targets = [[0, 1], [1, 0], [1, 0], [0, 1], [1, 0]]

shapes = [9, 7, 5, 3]
learning_rate = 0.1

input_size = (None, len(inputs[0]))
output_size = (None, len(targets[0]))

nn = sn.SimpleNet(
    input_shape=input_size,
    output_shape=output_size,
    hidden_layer_sizes=shapes,
    learning_rate=learning_rate,
    activation_function=sn.sigmoid,
    output_activation=sn.softmax,
    loss_function=sn.cross_entropy,
    dtype=dtype,
)

initial_weights, initial_biases = deepcopy(nn.weights), deepcopy(nn.biases)

keras_inputs = Input(shape=(len(inputs[0]),))

x = keras_inputs
for shape in [9, 7, 5, 3]:
    x = Dense(shape, activation="sigmoid")(x)
outputs = Dense(len(targets[0]), activation=None)(x)

model = Model(inputs=keras_inputs, outputs=outputs)
optimizer = SGD(learning_rate=learning_rate)
model.compile(
    optimizer=optimizer, loss=CategoricalCrossentropy(from_logits=True)
)

model.set_weights(
    [
        layer
        for weight, bias in zip(
            deepcopy(initial_weights), deepcopy(initial_biases)
        )
        for layer in (weight, bias.reshape(-1))
    ]
)

epochs = 10

for _ in range(epochs):
    nn.learn(inputs=inputs, targets=targets)

history = model.fit(
    np.array(inputs, dtype=dtype),
    np.array(targets, dtype=dtype),
    epochs=10,
    verbose=False,
)

pairs = {
    "weights": (model.get_weights()[::2], nn.weights),
    "biases": (model.get_weights()[1::2], nn.biases),
}


def test_learned_something() -> None:
    """Ensure that the weights and biases are changing."""
    for idx, (keras_data, nn_data) in enumerate(zip(*pairs["weights"])):
        assert not np.allclose(nn_data, initial_weights[idx])
        assert not np.allclose(keras_data, initial_weights[idx])
    for idx, (keras_data, nn_data) in enumerate(zip(*pairs["biases"])):
        assert not np.allclose(nn_data, initial_biases[idx])
        assert not np.allclose(keras_data, initial_biases[idx])


def test_weights_and_biases() -> None:
    """Ensure SimpleNet weights and biases match keras."""
    # `model.get_weights()` returns a list of alternating weights and biases

    for pair in pairs.values():
        for keras_data, nn_data in zip(*pair):
            assert np.allclose(keras_data, nn_data)


def test_err() -> None:
    """Ensure SimpleNet error matches keras."""
    assert np.isclose(nn.err, history.history["loss"][-1])
