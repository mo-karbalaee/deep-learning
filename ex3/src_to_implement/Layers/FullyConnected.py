import numpy as np
from Layers.Base import BaseLayer


class FullyConnected(BaseLayer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.trainable = True
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.uniform(0.0, 1.0, (input_size + 1, output_size))
        self._optimizer = None
        self._gradient_weights = None

    def initialize(self, weights_initializer, bias_initializer):
        w = weights_initializer.initialize(
            (self.input_size, self.output_size), self.input_size, self.output_size
        )
        b = bias_initializer.initialize((1, self.output_size), 1, self.output_size)
        self.weights = np.vstack((w, b))

    def forward(self, input_tensor):
        ones = np.ones((input_tensor.shape[0], 1))
        input_with_bias = np.hstack((input_tensor, ones))
        self.input_tensor = input_with_bias
        return input_with_bias @ self.weights

    def backward(self, error_tensor):
        self._gradient_weights = self.input_tensor.T @ error_tensor
        if self._optimizer:
            self.weights = self._optimizer.calculate_update(
                self.weights, self._gradient_weights
            )
        return error_tensor @ self.weights[:-1, :].T

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, opt):
        self._optimizer = opt

    @property
    def gradient_weights(self):
        return self._gradient_weights
