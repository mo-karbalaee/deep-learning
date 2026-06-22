import numpy as np

class Sgd:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def calculate_update(self, weight_tensor, gradient_tensor):
        return weight_tensor - self.learning_rate * gradient_tensor

class SgdWithMomentum:
    def __init__(self, learning_rate, momentum_rate):
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.v = None

    def calculate_update(self, weight_tensor, gradient_tensor):
        if self.v is None or self.v.shape != weight_tensor.shape:
            self.v = np.zeros_like(weight_tensor)
        self.v = self.momentum_rate * self.v - self.learning_rate * gradient_tensor
        return weight_tensor + self.v

class Adam:
    def __init__(self, learning_rate, mu, rho):
        self.learning_rate = learning_rate
        self.mu = mu
        self.rho = rho
        self.v = None
        self.r = None
        self.t = 0
        self.eps = 1e-8

    def calculate_update(self, weight_tensor, gradient_tensor):
        if self.v is None or self.v.shape != weight_tensor.shape:
            self.v = np.zeros_like(weight_tensor)
            self.r = np.zeros_like(weight_tensor)

        self.t += 1
        self.v = self.mu * self.v + (1.0 - self.mu) * gradient_tensor
        self.r = self.rho * self.r + (1.0 - self.rho) * (gradient_tensor ** 2)

        v_hat = self.v / (1.0 - self.mu ** self.t)
        r_hat = self.r / (1.0 - self.rho ** self.t)

        return weight_tensor - self.learning_rate * v_hat / (np.sqrt(r_hat) + self.eps)
