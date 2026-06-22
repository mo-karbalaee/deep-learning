import numpy as np

"""
As a reminder, the role of the optimizer in a neural network 
is to calculate the new weights in the backward pass. 
In the last exercise, we implemented stochastic gradient descent in the last exercise. 
This does nothing but going to the negative gradient tensor's direction. 
In this exercise, we implement two new optimizers that improve SGDs performance and 
address some of its flaws. 
"""


class Sgd:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def calculate_update(self, weight_tensor, gradient_tensor):
        return weight_tensor - self.learning_rate * gradient_tensor


"""
SGD with momentum is an attempt to make the update steps more adaptive to the 
shape of the field. This means that SGD with oscillate less on flat surfaces. How?
by adding a velocity term that takes the previous movements and the intensity of them 
into account for deciding the next move. It is like adding memory to SGD. 
Makes SGD less stochastic and zig-zagy. 
"""


class SgdWithMomentum:
    """
    In this method, the learning rate is still constant and does not adapt.
    The constructor takes the constant learning rate and also the constant momentum rate.
    """

    def __init__(self, learning_rate, momentum_rate):
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        """
        Of course the velocity is null in the beginning. 
        """
        self.v = None

    def calculate_update(self, weight_tensor, gradient_tensor):
        """
        Ok so first of all, if it is the first iteration, then there is no velocity, hence
        we need to initialize it with a tensor. Also when the shape of the velocity and the
        weight tensor is not the same, it means that the velocity belongs to the previous layer and
        it needs to be reset.
        """
        if self.v is None or self.v.shape != weight_tensor.shape:
            """
            Makes a tensor of the same size as the weight tensor filled with zeros.
            The reason that the shape of the velocity is the same as the weight tensor is that
            every weight has its own velocity.
            """
            self.v = np.zeros_like(weight_tensor)
        """
        This is just updating the velocity for the next step. 
        """
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
        self.r = self.rho * self.r + (1.0 - self.rho) * (gradient_tensor**2)

        v_hat = self.v / (1.0 - self.mu**self.t)
        r_hat = self.r / (1.0 - self.rho**self.t)

        return weight_tensor - self.learning_rate * v_hat / (np.sqrt(r_hat) + self.eps)
