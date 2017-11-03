"""test_simplenet.py :: Tests for `simplenet` module."""

import numpy as np

import simplenet as sn


def test_validate_neg_log_likelihood() -> None:
    """Use gradient checking to validate neg_log_likelihood cost."""
    inputs = [[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1], [1, 1, 0]]
    targets = [[0], [1], [1], [0], [1]]

    shapes = [9, 7, 5, 3]

    nn = sn.SimpleNet(
        input_shape=(None, len(inputs[0])),
        output_shape=(None, len(targets[0])),
        hidden_layer_sizes=shapes,
        learning_rate=0.1,
        activation_function=sn.sigmoid,
        output_activation=sn.sigmoid,
        loss_function=sn.neg_log_likelihood,
        dtype=np.float64,
        seed=42,
        )

    assert nn.validate(inputs=inputs, targets=targets, epsilon=1e-7)


def test_validate_cross_entropy() -> None:
    """Use gradient checking to validate cross_entropy cost."""
    inputs = [[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1], [1, 1, 0]]
    targets = [[0, 1], [1, 0], [1, 0], [0, 1], [1, 0]]

    shapes = [9, 7, 5, 3]

    nn = sn.SimpleNet(
        input_shape=(None, len(inputs[0])),
        output_shape=(None, len(targets[0])),
        hidden_layer_sizes=shapes,
        learning_rate=0.1,
        activation_function=sn.sigmoid,
        output_activation=sn.sigmoid,
        loss_function=sn.neg_log_likelihood,
        dtype=np.float64,
        seed=42,
        )

    assert nn.validate(inputs=inputs, targets=targets, epsilon=1e-7)
