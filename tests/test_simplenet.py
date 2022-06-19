"""test_simplenet.py :: Tests for `simplenet` module."""

import tempfile

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
        dtype="float64",
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
        dtype="float64",
        seed=42,
    )

    assert nn.validate(inputs=inputs, targets=targets, epsilon=1e-7)


def test_import_export() -> None:
    """Test that I can export and re-import weights."""
    dtype = "float64"
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

    nn2 = sn.SimpleNet(
        input_shape=input_size,
        output_shape=output_size,
        hidden_layer_sizes=shapes,
        learning_rate=learning_rate,
        activation_function=sn.sigmoid,
        output_activation=sn.softmax,
        loss_function=sn.cross_entropy,
        dtype=dtype,
    )

    for idx, weight in enumerate(nn.weights):
        assert not np.allclose(weight, nn2.weights[idx])

    with tempfile.NamedTemporaryFile(suffix=".npz") as f:
        nn.export_model(f.name)
        nn2.import_model(f.name)

    for idx, weight in enumerate(nn.weights):
        assert np.allclose(weight, nn2.weights[idx])

    for idx, bias in enumerate(nn.biases):
        assert np.allclose(bias, nn2.biases[idx])
