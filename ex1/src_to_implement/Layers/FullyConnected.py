from Layers.Base import BaseLayer
import numpy as np

class FullyConnected(BaseLayer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.trainable = True
        self._optimizer = None
        self._gradient_weights = None
        self.weights = np.random.uniform(0, 1, (input_size + 1, output_size))

    def forward(self, input_tensor):
        self.input_tensor = np.hstack([input_tensor, np.ones((input_tensor.shape[0], 1))])
        return self.input_tensor @ self.weights

    def backward(self, error_tensor):
        prev_error_tensor = error_tensor @ self.weights.T
        self._gradient_weights = self.input_tensor.T @ error_tensor
        if self._optimizer is not None:
            self.weights = self._optimizer.calculate_update(self.weights, self._gradient_weights)
        return prev_error_tensor[:, :-1]

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, value):
        self._optimizer = value

    @property
    def gradient_weights(self):
        return self._gradient_weights