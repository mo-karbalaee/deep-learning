import numpy as np
from Layers.Base import BaseLayer

class Flatten(BaseLayer):
    def __init__(self):
        super().__init__()
        self.trainable = False
        self.input_shape = None

    def forward(self, input_tensor):
        """
        We save the input shape because we want to use it in the backward path
        to reshape it back to the feature map and give it to the conv layers.
        """
        self.input_shape = input_tensor.shape
        batch_size = input_tensor.shape[0]
        return input_tensor.reshape(batch_size, -1)

    def backward(self, error_tensor):
        return error_tensor.reshape(self.input_shape)
