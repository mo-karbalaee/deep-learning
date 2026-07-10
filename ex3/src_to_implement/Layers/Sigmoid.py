import numpy as np
from Layers.Base import BaseLayer


class Sigmoid(BaseLayer):

    def __init__(self):
        super().__init__()
        self.trainable = False
        self.activation = None

    def forward(self, input_tensor):
        self.activation = 1.0 / (1.0 + np.exp(-input_tensor))
        return self.activation

    def backward(self, error_tensor):
        return error_tensor * self.activation * (1.0 - self.activation)
