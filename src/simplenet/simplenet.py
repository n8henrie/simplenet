"""simplenet.simplenet :: Define SimpleNet class and common functions."""

from typing import Callable, List, Sequence, Tuple, Union  # noqa

import numpy as np


DataArray = Union[Sequence[int], Sequence[float], np.ndarray]


def sigmoid(arr: np.ndarray, der: bool = False) -> np.array:
    r"""Calculate the sigmoid activation function.

    .. math::
        \frac{1}{1 + e ^ {-x}}

    Derivative:

    .. math::
        x * (1 - x)

    Args:
        arr: Input array of weighted sums
    Returns:
        Array of outputs from 0 to 1
    """
    activations = 1 / (1 + np.exp(-arr))
    if der is True:
        return activations * (1 - activations)
    return activations


def softmax(arr: np.ndarray) -> np.ndarray:
    r"""Calculate the softmax activation function.

    This equation uses a "stable softmax" that subtracts the maximum from the
    exponents, but which should not change the results.

    .. math::
        \frac{e^x}{\sum_{} {e^x}}

    Args:
        arr: Input array of weighted sums
    Returns:
        Array of outputs from 0 to 1
    """
    exps = np.exp(arr - np.max(arr, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)


def neg_log_likelihood(y_hat: np.ndarray, targets: np.ndarray,
                       der: bool = False) -> float:
    r"""Calculate the negative log likelihood loss.

    I believe this is also called the binary cross-entropy loss function.

    Args:
        y_hat: Array of predicted values from 0 to 1
        targets: Array of true values
    Returns:
        Mean loss for the sample
    """
    m = y_hat.shape[0]

    if der is True:
        return (1 / m) * (y_hat - targets)
    return -(1 / m) * np.sum(
            targets * np.log(y_hat) + (1 - targets) * np.log(1 - y_hat)
            )


def cross_entropy(y_hat: np.ndarray, targets: np.ndarray,
                  der: bool = False) -> float:
    """Calculate the categorical cross entropy loss.

    Args:
        y_hat: Array of predicted values from 0 to 1
        targets: Array of true values
    Returns:
        Mean loss for the sample
    """
    m = y_hat.shape[0]

    if der is True:
        return (1 / m) * (y_hat - targets)
    return -(1 / m) * np.sum(targets * np.log(y_hat))


def relu(arr: np.ndarray, der: bool = False) -> np.ndarray:
    """Calculate the relu activation function.

    Args:
        arr: Input array
        der: Whether to calculate the derivative
    Returns:
        Array of outputs from 0 to maximum of the array in a given axis
    """
    if der is True:
        return np.where(arr <= 0, 0, 1)
    return np.maximum(arr, 0)


class SimpleNet:
    """Simple example of a multilayer perceptron."""

    def __init__(
        self,
        hidden_layer_sizes: Sequence[int],
        input_shape: Tuple[int, int],
        output_shape: Tuple[int, int],
        activation_function: Callable[..., np.ndarray] = sigmoid,
        output_activation: Callable[..., np.ndarray] = sigmoid,
        loss_function: Callable[..., float] = neg_log_likelihood,
        learning_rate: float = 1.,
        dtype: str = 'float32',
        seed: int = None,
    ) -> None:
        """Initialize the MPL.

        Args:
            hidden_layer_sizes: Number of neurons in each hidden layer
            input_shape: Shape of inputs (m x n), use `None` for unknown m
            output_shape: Shape of outputs (m x o), use `None` for unknown m
            activation_function: Activation function for all layers prior to
                                 output
            output_activation: Activation function for output layer
            learning_rate: learning rate
            dtype: Data type for floats (e.g. np.float32 vs np.float64)
            seed: Optional random seed for consistent outputs (for debugging)
        """
        self.dtype = dtype
        np.random.seed(seed=seed)
        layer_sizes = ([input_shape[1]] + list(hidden_layer_sizes) +
                       [output_shape[1]])

        self.weights = [
                np.random.normal(size=(layer_size, next_layer_size),
                                 scale=0.01
                                 ).astype(self.dtype)
                for layer_size, next_layer_size in
                zip(layer_sizes, layer_sizes[1:])
            ]

        self.zs = [np.full((size, 1), np.nan, dtype=self.dtype)
                   for size in layer_sizes[1:]]
        self.outputs = [z.copy() for z in self.zs]
        self.biases = [np.zeros((1, layer_size), dtype=self.dtype)
                       for layer_size in layer_sizes[1:]]

        self.activation_function = activation_function
        self.learning_rate = learning_rate
        self.output_activation = output_activation
        self.loss_function = loss_function

    def _forward(self, inputs: np.ndarray) -> None:
        """Perform the forward pass.

        Args:
            inputs: Array of input values
        """
        self.zs[0] = np.dot(inputs, self.weights[0]) + self.biases[0]
        self.outputs[0] = self.activation_function(self.zs[0])

        for layer_num in range(1, len(self.weights)):
            self.zs[layer_num] = np.dot(self.outputs[layer_num - 1],
                                        self.weights[layer_num]) + \
                                 self.biases[layer_num]

            if layer_num < len(self.weights) - 1:
                self.outputs[layer_num] = self.activation_function(
                    self.zs[layer_num])
            else:
                self.outputs[layer_num] = self.output_activation(
                    self.zs[layer_num])

    def _backprop(self, inputs: np.ndarray, targets: np.ndarray) -> None:
        """Calculate gradients and perform the backward pass.

        Args:
            inputs: Array of input values
            targets: Array of true outputs
        """
        y_hat = self.outputs[-1]

        self.err = self.loss_function(y_hat=y_hat, targets=targets)

        dws = []  # type: List[np.ndarray]
        dbs = []  # type: List[np.ndarray]
        dzs = [self.loss_function(y_hat=y_hat, targets=targets, der=True)]

        for output, weight, z in zip(self.outputs[-2::-1],
                                     self.weights[::-1],
                                     self.zs[-2::-1]):
            dws.insert(0, np.dot(output.T, dzs[0]))
            dbs.insert(0, np.sum(dzs[0], axis=0, keepdims=True))

            dzs.insert(0, np.dot(dzs[0], weight.T) *
                       self.activation_function(z, der=True))

        dws.insert(0, np.dot(inputs.T, dzs[0]))
        dbs.insert(0, np.sum(dzs[0], axis=0, keepdims=True))

        for idx, (dw, db) in enumerate(zip(dws, dbs)):
            self.weights[idx] -= self.learning_rate * dw
            self.biases[idx] -= self.learning_rate * db

    def learn(self, inputs: DataArray, targets: DataArray) -> None:
        """Perform a forward and backward pass, updating weights.

        Args:
            inputs: Array of input values
            targets: Array of true outputs
        """
        inputs = np.array(inputs, dtype=self.dtype)
        targets = np.array(targets, dtype=self.dtype)
        self._forward(inputs=inputs)
        self._backprop(inputs=inputs, targets=targets)

    def predict(self, inputs: DataArray) -> np.ndarray:
        """Use existing weights to predict outputs for given inputs.

        Note: this method does *not* update weights.

        Args:
            inputs: Array of inputs for which to make predictions
        Returns:
            Array of predictions
        """
        inputs = np.array(inputs, dtype=self.dtype)

        zs = [z.copy() for z in self.zs]
        outputs = [output.copy() for output in self.outputs]

        zs[0] = np.dot(inputs, self.weights[0]) + self.biases[0]
        outputs[0] = self.activation_function(zs[0])

        for layer_num in range(1, len(self.weights)):
            zs[layer_num] = np.dot(outputs[layer_num - 1],
                                   self.weights[layer_num]) + \
                            self.biases[layer_num]

            if layer_num < len(self.weights) - 1:
                outputs[layer_num] = self.activation_function(
                    zs[layer_num])
            else:
                outputs[layer_num] = self.output_activation(
                    zs[layer_num])

        return outputs[-1]

    def validate(self, inputs: np.ndarray, targets: np.ndarray,
                 epsilon: float = 1e-7) -> bool:
        """Use gradient checking to validate backpropagation.

        This method uses a naive implementation of gradient checking to try to
        verify the analytic gradients.

        Args:
            inputs: Array of input values
            targets: Array of true outputs
            epsilon: Small value by which to perturb values for gradient
                     checking
        Returns:
            Boolean reflecting whether or not the gradients seem to match
        """
        targets_arr = np.array(targets, dtype=self.dtype)

        weight_grads = []  # type: List[List[List[float]]]
        bias_grads = []  # type: List[List[List[float]]]

        backup_weights = [weight.copy() for weight in self.weights]
        backup_biases = [bias.copy() for bias in self.biases]

        for layer_num, layer_weights in enumerate(self.weights):
            layer_weight_grads = []  # type: List[List[float]]
            layer_bias_grads = [[]]  # type: List[List[float]]

            for neuron_num, neuron_weights in enumerate(layer_weights):

                neuron_weight_grads = []

                for weight_num, weight in enumerate(neuron_weights):
                    self.weights[layer_num][neuron_num][weight_num] = \
                        weight + epsilon
                    outputs = self.predict(inputs)
                    cost_plus = self.loss_function(y_hat=outputs,
                                                   targets=targets_arr)

                    self.weights[layer_num][neuron_num][weight_num] = \
                        weight - epsilon
                    outputs = self.predict(inputs)
                    cost_minus = self.loss_function(y_hat=outputs,
                                                    targets=targets_arr)

                    self.weights = [backup_weight.copy()
                                    for backup_weight in backup_weights]
                    weight_grad = (cost_plus - cost_minus) / (2 * epsilon)
                    neuron_weight_grads.append(weight_grad)

                    # Biases are shape (1, len(next_layer)), and
                    # len(next_layer) == len(neuron_weights) so only set biases
                    # once per neuron, using the neuron's weight_num to index
                    if neuron_num == 0:
                        bias = self.biases[layer_num][0][weight_num]

                        self.biases[layer_num][0][weight_num] = bias + epsilon
                        outputs = self.predict(inputs)
                        cost_plus = self.loss_function(y_hat=outputs,
                                                       targets=targets_arr)

                        self.biases[layer_num][0][weight_num] = bias - epsilon
                        outputs = self.predict(inputs)
                        cost_minus = self.loss_function(y_hat=outputs,
                                                        targets=targets_arr)

                        self.biases = [backup_bias.copy()
                                       for backup_bias in backup_biases]
                        neuron_bias_grad = (cost_plus - cost_minus) / \
                            (2 * epsilon)
                        layer_bias_grads[0].append(neuron_bias_grad)

                layer_weight_grads.append(neuron_weight_grads)

            weight_grads.append(layer_weight_grads)
            bias_grads.append(layer_bias_grads)

        self.learn(inputs=inputs, targets=targets)

        weight_deltas = []
        bias_deltas = []

        for weight_before, weight_after, bias_before, bias_after in \
                zip(backup_weights, self.weights, backup_biases, self.biases):
            weight_deltas.append(
                    (weight_before - weight_after) / self.learning_rate)
            bias_deltas.append((bias_before - bias_after) / self.learning_rate)

        self.weights = [backup_weight.copy()
                        for backup_weight in backup_weights]
        self.biases = [backup_bias.copy() for backup_bias in backup_biases]

        pairs = {
                "weight": (weight_grads, weight_deltas),
                "bias": (bias_grads, bias_deltas),
                }
        for k, pair in pairs.items():
            for idx, (calculated, analytic) in enumerate(zip(*pair)):
                if not np.allclose(calculated, analytic):
                    width = 25
                    print("Wrong {} gradient suspected around layer {}."
                          .format(k, idx))
                    header = ("{'calculated':^{width}}"
                              "{'analytic':^{width}}"
                              "{'diff':^{width}}")
                    print(header.format(width=width))

                    for c, a in zip(np.array(calculated).reshape(-1),
                                    analytic.reshape(-1)):
                        print("{:^{width}}{a:^{width}}{c-a:^{width}}"
                              .format(c, a, c-a, width=width))

                    return False

        print("All {} gradients check out.".format(k))
        return True

    def export_model(self, filename: str) -> None:
        """Export the learned biases and weights to a file.

        Saves each weight and bias in order with an index and a prefix of `W`
        or `b` to ensure it can be restored in the proper order.

        Args:
            filename: Filename for the saved file.
        """
        pad = len(str(len(self.weights)))
        weights = {"W{:0{pad}}".format(idx, pad=pad): self.weights[idx]
                   for idx in range(len(self.weights))}
        biases = {"b{:0{pad}}".format(idx, pad=pad): self.weights[idx]
                  for idx in range(len(self.weights))}
        np.savez(filename, **weights, **biases)

    def import_model(self, filename: str) -> None:
        """Import learned biases and weights from a file.

        Args:
            filename: Name of file from which to import
        """
        model = np.load(filename)
        self.weights = [model[k] for k in sorted(model.keys())
                        if k.startswith("W")]
        self.biases = [model[k] for k in sorted(model.keys())
                       if k.startswith("b")]
