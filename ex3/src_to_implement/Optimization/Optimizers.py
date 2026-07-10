import numpy as np

"""
As a reminder, the role of the optimizer in a neural network
is to calculate the new weights in the backward pass.

For exercise 3 we introduce a common base class 'Optimizer' so that every optimizer
can optionally hold a regularizer. If a regularizer is present, the optimizer first
'shrinks' the weights by subtracting learning_rate * regularizer.calculate_gradient(w)
and only then performs its usual update step with the data gradient.
"""


class Optimizer:
    """
    Base optimizer. Stores an optional regularizer that all derived optimizers can
    apply at the beginning of the weight update.
    """

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
            # Shrink the weights first, then do the normal gradient step.
            weight_tensor = weight_tensor - self.learning_rate * self.regularizer.calculate_gradient(weight_tensor)
        return weight_tensor - self.learning_rate * gradient_tensor


"""
SGD with momentum is an attempt to make the update steps more adaptive to the
shape of the field. This means that SGD with oscillate less on flat surfaces. How?
by adding a velocity term that takes the previous movements and the intensity of them
into account for deciding the next move. It is like adding memory to SGD.
Makes SGD less stochastic and zig-zagy.
"""


class SgdWithMomentum(Optimizer):
    """
    In this method, the learning rate is still constant and does not adapt.
    The constructor takes the constant learning rate and also the constant momentum rate.
    """

    def __init__(self, learning_rate, momentum_rate):
        super().__init__()
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        """
        Of course the velocity is null in the beginning.
        """
        self.v = None

    def calculate_update(self, weight_tensor, gradient_tensor):
        if self.regularizer is not None:
            # Apply the regularizer shrinkage to the weights before the momentum step.
            weight_tensor = weight_tensor - self.learning_rate * self.regularizer.calculate_gradient(weight_tensor)
        """
        Ok so first of all, if it is the first iteration, then there is no velocity, hence
        we need to initialize it with a tensor. Also when the shape of the velocity and the
        weight tensor is not the same, it means that the velocity belongs to the previous layer and
        it needs to be reset.
        """
        if self.v is None or self.v.shape != weight_tensor.shape:
            self.v = np.zeros_like(weight_tensor)
        """
        This is just updating the velocity for the next step.
        """
        self.v = self.momentum_rate * self.v - self.learning_rate * gradient_tensor
        return weight_tensor + self.v


"""
Adam is yet another optimization algorithm that combines SGD with momentum to RMSprop.
We learnt SGD with momentum, but what is RMSprop?
Root mean squared propagation. The idea of RMSprop is to adapt the learning for each parameter
in the network based on the magnitude of its gradients. It means that the learning rate will not be
constant and will be adaptive.
"""


class Adam(Optimizer):
    """
    The learning rate is received in the constructor as the initial value globally in the network
    then Adam adapts it per parameter.
    In short, mu is the decay rate for the first moment v and rho is the decay of the second moment.
    """

    def __init__(self, learning_rate, mu, rho):
        super().__init__()
        self.learning_rate = learning_rate
        self.mu = mu
        self.rho = rho
        self.v = None
        self.r = None
        """
        Step counter. Used for bias correction. Increments every time calculate_update is called.
        """
        self.t = 0
        """
        Tiny constant to avoid division by zero later.
        """
        self.eps = 1e-8

    def calculate_update(self, weight_tensor, gradient_tensor):
        if self.regularizer is not None:
            # Regularizer shrinkage applied before the adaptive moment update.
            weight_tensor = weight_tensor - self.learning_rate * self.regularizer.calculate_gradient(weight_tensor)
        if self.v is None or self.v.shape != weight_tensor.shape:
            self.v = np.zeros_like(weight_tensor)
            self.r = np.zeros_like(weight_tensor)

        self.t += 1
        self.v = self.mu * self.v + (1.0 - self.mu) * gradient_tensor
        self.r = self.rho * self.r + (1.0 - self.rho) * (gradient_tensor ** 2)

        v_hat = self.v / (1.0 - self.mu ** self.t)
        r_hat = self.r / (1.0 - self.rho ** self.t)

        return weight_tensor - self.learning_rate * v_hat / (np.sqrt(r_hat) + self.eps)
