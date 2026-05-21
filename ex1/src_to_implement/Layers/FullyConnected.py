from Base import BaseLayer
import numpy as np

class FullyConnected(BaseLayer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.trainable = True
        self.weights = np.random.uniform(0, 1, (input_size, output_size))

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        return np.matmul(input_tensor, self.weights)

    def backward(self, error_tensor):
        if self.optimizer == None:
            raise ValueError("Optimizer is missing!")
        
        prev_error_tensor = error_tensor @ self.weights.T
        gradient_tensor = self.input_tensor.T @ error_tensor
        self.weights = self.optimizer.calculate_update(self.weights, gradient_tensor)
        return prev_error_tensor
    
    @property
    def optimizer(self):
        return self._optimizer


    @optimizer.setter
    def optimizer(self, value):
        self._optimizer = value

    
    @property
    def gradient_weights(self):
        return self._gradient_weights


    @gradient_weights.setter
    def gradient_weights(self, value):
        self._gradient_weights = value           