"""Compare an update against an identical network in Tensorflow."""

import numpy as np
import tensorflow as tf

import simplenet as sn

inputs = [[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1], [1, 1, 0]]
targets = [[0, 1], [1, 0], [1, 0], [0, 1], [1, 0]]

shapes = [9, 7, 5, 3]
learning_rate = 0.1

input_size = (None, len(inputs[0]))
output_size = (None, len(targets[0]))

nn = sn.SimpleNet(
    input_shape=input_size, output_shape=output_size,
    hidden_layer_sizes=shapes,
    learning_rate=learning_rate,
    activation_function=sn.sigmoid,
    output_activation=sn.softmax,
    loss_function=sn.cross_entropy,
    dtype="float64",
    )

W0, W1, W2, W3, W4 = [tf.Variable(weight, dtype=nn.dtype)
                      for weight in nn.weights]

b0, b1, b2, b3, b4 = [tf.Variable(tf.zeros([shape], dtype=nn.dtype))
                      for shape in shapes + [output_size[1]]]

X0 = tf.placeholder(nn.dtype, input_size)
y = tf.placeholder(nn.dtype, output_size)

X1 = tf.nn.sigmoid(X0 @ W0 + b0)
X2 = tf.nn.sigmoid(X1 @ W1 + b1)
X3 = tf.nn.sigmoid(X2 @ W2 + b2)
X4 = tf.nn.sigmoid(X3 @ W3 + b3)
logits = X4 @ W4 + b4

cost = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits))
train_step = tf.train.GradientDescentOptimizer(nn.learning_rate).minimize(cost)

nn.learn(inputs=inputs, targets=targets)

with tf.Session() as sess:
    tf.global_variables_initializer().run()

    _, tc = sess.run([train_step, cost], feed_dict={X0: inputs, y: targets})

    pairs = {
        'weights': (sess.run([W0, W1, W2, W3, W4]), nn.weights),
        'biases': (sess.run([b0, b1, b2, b3, b4]), nn.biases),
    }


def test_weights_and_biases() -> None:
    """Ensure SimpleNet weights and biases match Tensorflow after 1 epoch."""
    for k, pair in pairs.items():
        for idx, (tf_data, nn_data) in enumerate(zip(*pair)):
            assert np.allclose(tf_data, nn_data)


def test_err() -> None:
    """Ensure SimpleNet error matches Tensorflow after 1 epoch."""
    assert np.isclose(nn.err, tc)


def test_softmax() -> None:
    """Test implementation of stable softmax vs Tensorflow."""
    arr = np.random.random((3, 4)) * 10
    with tf.Session() as sess:
        assert np.allclose(sn.softmax(arr), sess.run(tf.nn.softmax(arr)))
