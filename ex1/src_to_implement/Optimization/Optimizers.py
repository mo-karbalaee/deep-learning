import numpy as np

class Sgd:
    def __init__(self, learning_rate:float):
        self.learning_rate = learning_rate

    def calculate_update(weight_tensor: np.ndarray, gradient_tensor: np.ndarray) -> np.ndarray:
        pass
