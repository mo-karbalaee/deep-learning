import numpy as np

"""
One of the key changes have been here. We implemented a parent class 
and made all the previously-implemented optimizers extend it so that we
don't have to copy and paste optimizer code for each of these classes. 
They inherit that from their parent class. 
"""
class Optimizer:

    def __init__(self):
        self.regularizer = None

    def add_regularizer(self, regularizer):
        self.regularizer = regularizer


class Sgd(Optimizer):
    def __init__(self, learning_rate):
        super().__init__()
        self.learning_rate = learning_rate

    def calculate_update(self, weight_tensor, gradient_tensor):
        if self.regularizer is not None:
            weight_tensor = (
                weight_tensor
                - self.learning_rate
                * self.regularizer.calculate_gradient(weight_tensor)
            )
        return weight_tensor - self.learning_rate * gradient_tensor


class SgdWithMomentum(Optimizer):

    def __init__(self, learning_rate, momentum_rate):
        super().__init__()
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.v = None

    def calculate_update(self, weight_tensor, gradient_tensor):
        if self.regularizer is not None:
            weight_tensor = (
                weight_tensor
                - self.learning_rate
                * self.regularizer.calculate_gradient(weight_tensor)
            )
        if self.v is None or self.v.shape != weight_tensor.shape:
            self.v = np.zeros_like(weight_tensor)
        self.v = self.momentum_rate * self.v - self.learning_rate * gradient_tensor
        return weight_tensor + self.v


class Adam(Optimizer):

    def __init__(self, learning_rate, mu, rho):
        super().__init__()
        self.learning_rate = learning_rate
        self.mu = mu
        self.rho = rho
        self.v = None
        self.r = None
        self.t = 0
        self.eps = 1e-8

    def calculate_update(self, weight_tensor, gradient_tensor):
        if self.regularizer is not None:
            weight_tensor = (
                weight_tensor
                - self.learning_rate
                * self.regularizer.calculate_gradient(weight_tensor)
            )
        if self.v is None or self.v.shape != weight_tensor.shape:
            self.v = np.zeros_like(weight_tensor)
            self.r = np.zeros_like(weight_tensor)

        self.t += 1
        self.v = self.mu * self.v + (1.0 - self.mu) * gradient_tensor
        self.r = self.rho * self.r + (1.0 - self.rho) * (gradient_tensor**2)

        v_hat = self.v / (1.0 - self.mu**self.t)
        r_hat = self.r / (1.0 - self.rho**self.t)

        return weight_tensor - self.learning_rate * v_hat / (np.sqrt(r_hat) + self.eps)
