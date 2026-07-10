import numpy as np
from Layers.Base import BaseLayer


class SoftMax(BaseLayer):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        input_stable = input_tensor - np.max(input_tensor, axis=1, keepdims=True)
        exp_tensor = np.exp(input_stable)
        self.output_tensor = exp_tensor / np.sum(exp_tensor, axis=1, keepdims=True)
        return self.output_tensor

    def backward(self, error_tensor):
        batch_size, num_classes = self.output_tensor.shape
        gradient = np.zeros_like(self.input_tensor)
        for b in range(batch_size):
            y = self.output_tensor[b : b + 1, :]
            jacobian = np.diag(y.flatten()) - np.outer(y.flatten(), y.flatten())
            gradient[b, :] = error_tensor[b, :] @ jacobian
        return gradient
