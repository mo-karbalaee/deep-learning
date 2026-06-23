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

"""
Adam is yet another optimization algorithm that combines SGD with momentum to RMSprop. 
We learnt SGD with momentum, but what is RMSprop?
Root mean squared propagation. The idea of RMSprop is to adapt the learning for each parameter
in the network based on the magnitude of its gradients. It means that the learning rate will not be 
constant and will be adaptive. 
"""
class Adam:
    """
    The learning rate is received in the constructor as the initial value globally in the network
    then Adam adapts it per parameter. 
    mu and rho are hyper parameters of the Adam optimizer. Both are a value between zero and one. 
    mu specifies how much of the first moment is kept. If mu is 0.9, 90 percent of the previous moves are
    kept and used for determining the next direction. 
    In short, mu is the decay rate for the first moment v and rho is the decay of the second moment. 
    مو 
    رو
    """
    def __init__(self, learning_rate, mu, rho):
        self.learning_rate = learning_rate
        self.mu = mu
        self.rho = rho
        """
        These are running averages. 
        In statistics, a moment is a measure that describes the shape of a distribution.
        - **First moment** — the mean. The average value. Tells you the central tendency/direction.
        - **Second moment** — the average of squared values. Tells you the magnitude/spread.

        In Adam it's the same idea applied to gradients over time instead of a distribution:

        - **First moment** — average of past gradients → direction
        - **Second moment** — average of past squared gradients → magnitude
        """
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
        """
        Exactly like the previous optimizer above, we have initialized the moments with null at first
        since we didn't know the shape of the weight tensor and then we initialized. And if the shapes don't match
        it means that the moments belong to the previous layer and that they should be reinitialized. 
        """
        if self.v is None or self.v.shape != weight_tensor.shape:
            """
            The moments have the same shape as the weight tensors because each weight has its
            own moment estimate. So there is a one-to-one mapping between them. 
            """
            self.v = np.zeros_like(weight_tensor)
            self.r = np.zeros_like(weight_tensor)

        """
        We keep record of the step because it is used in the formula. 
        """
        self.t += 1
        """
        This is the intuition. 
        Instead of using only the current gradient, Adam remembers recent gradients.
        """
        self.v = self.mu * self.v + (1.0 - self.mu) * gradient_tensor
        """
        We use the square here because we care about the magnitude of the gradients here instead of 
        the direction. 
        """
        self.r = self.rho * self.r + (1.0 - self.rho) * (gradient_tensor**2)
        
        """
        This is the bias correction step. 
        """
        v_hat = self.v / (1.0 - self.mu**self.t)
        r_hat = self.r / (1.0 - self.rho**self.t)
        
        """
        This is the final line that updates the weights. Mention the use of eps here. 
        """
        return weight_tensor - self.learning_rate * v_hat / (np.sqrt(r_hat) + self.eps)
