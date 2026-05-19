import numpy as np

class Sgd:
    def __init__(self, learning_rate:float):
        self.learning_rate = learning_rate

    def calculate_update(self, weight_tensor: np.ndarray, gradient_tensor: np.ndarray) -> np.ndarray:
           assert weight_tensor.shape == gradient_tensor.shape, \
        f"Shape mismatch error: Weights {weight_tensor.shape} vs Gradients {gradient_tensor.shape}"
           
           new_weight_tensor = weight_tensor - (self.learning_rate * gradient_tensor)

           return new_weight_tensor
