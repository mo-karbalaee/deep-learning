from Base import BaseLayer
import numpy as np

class SoftMax(BaseLayer):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor):
        shifted = input_tensor - np.max(input_tensor, axis=1, keepdims=True)
        exp = np.exp(shifted)
        self.output = exp / np.sum(exp, axis=1, keepdims=True)
        return self.output

    def backward(self, error_tensor):
        return self.output * (error_tensor - np.sum(error_tensor * self.output, axis=1, keepdims=True))