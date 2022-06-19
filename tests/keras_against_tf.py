"""Ensure sure keras and tf give the same results as a sanity test."""

from copy import deepcopy

import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow import keras
from tensorflow.python.keras.layers import Dense, Input
from tensorflow.python.keras.losses import CategoricalCrossentropy
from tensorflow.python.keras.models import Model

import simplenet as sn

dtype = "float64"
tf.disable_v2_behavior()
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

nn.import_model("nn.npz")
initial_weights, initial_biases = deepcopy(nn.weights), deepcopy(nn.biases)

W0, W1, W2, W3, W4 = [
    tf.Variable(weight, dtype=dtype) for weight in deepcopy(initial_weights)
]

b0, b1, b2, b3, b4 = [
    tf.Variable(bias, dtype=dtype) for bias in deepcopy(initial_biases)
]


X0 = tf.placeholder(dtype, input_size)
y = tf.placeholder(dtype, output_size)

X1 = tf.nn.sigmoid(X0 @ W0 + b0)
X2 = tf.nn.sigmoid(X1 @ W1 + b1)
X3 = tf.nn.sigmoid(X2 @ W2 + b2)
X4 = tf.nn.sigmoid(X3 @ W3 + b3)
logits = X4 @ W4 + b4

cost = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits),
)
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

epochs = 10

with tf.Session() as sess:
    with sess.as_default():
        tf.global_variables_initializer().run()
        for _ in range(epochs):
            _, tc = sess.run(
                [train_step, cost],
                feed_dict={X0: inputs, y: targets},
            )

        tf_vals = {
            "weights": sess.run([W0, W1, W2, W3, W4]),
            "biases": sess.run([b0, b1, b2, b3, b4]),
        }


keras_inputs = Input(shape=(len(inputs[0]),))

x = keras_inputs
for shape in [9, 7, 5, 3]:
    x = Dense(shape, activation="sigmoid")(x)
outputs = Dense(len(targets[0]), activation=None)(x)

model = Model(inputs=keras_inputs, outputs=outputs)
optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
model.compile(optimizer="SGD", loss=CategoricalCrossentropy(from_logits=True))

model.set_weights(
    [
        layer
        for weight, bias in zip(
            deepcopy(initial_weights), deepcopy(initial_biases)
        )
        for layer in (weight, bias.reshape(-1))
    ]
)

history = model.fit(
    np.array(deepcopy(inputs)),
    np.array(deepcopy(targets)),
    epochs=epochs,
    verbose="0",
)


keras_vals = {
    "weights": model.get_weights()[::2],
    "biases": model.get_weights()[1::2],
}


def test_learned_something() -> None:
    """Ensure that the weights and biases are changing."""
    for key in ["weights", "biases"]:
        assert len(keras_vals[key]) == len(tf_vals[key])
        for idx in range(len(initial_weights)):
            assert not np.allclose(keras_vals[key][idx], initial_weights[idx])
            assert not np.allclose(tf_vals[key][idx], initial_weights[idx])


def test_weights_and_biases() -> None:
    """Ensure tf weights and biases match keras."""
    for k in ["weights", "biases"]:
        for idx in range(len(keras_vals[k])):
            assert np.allclose(keras_vals[k][idx], tf_vals[k][idx])


def test_err() -> None:
    """Ensure tf error matches keras."""
    assert np.isclose(history.history["loss"][-1], tc)
