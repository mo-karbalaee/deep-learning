import numpy as np

"""
This file contains all the classes and functions that can help with weight 
initializations. We know that there are different methods for initialization of the 
weights of the networks and each have their own pros and cons. 
"""

"""
The first type of initialization is constant initialization where 
all the weights of the network are set to a constant value. 
Everything to the same value. 
"""


class Constant:
    """
    Class constructor that receives the constant_value parameter which is the value
    that we want to put as the weights of the network. It falls back to 0.1 if not passed.
    """

    def __init__(self, constant_value=0.1):
        self.constant_value = constant_value

    """
    These initialization functions are supposed to return the initial weight tensors
    of the network. 
    # parameters
    1. weights_shape: The shape of the weight tensor. We don't need the tensor itself
        we just need the shape, we'll do the initialization and everything in the function itself. 
    2. fan_in: How many neurons feed into a single neuron in the current layer.
    3. fan_out: How many neurons does a neuron in the current layer feed into the next layer.     
    """

    def initialize(self, weights_shape, fan_in, fan_out):
        """
        np.full makes a tensor of the requested shape filled with the specified value.
        """
        return np.full(weights_shape, self.constant_value)


class UniformRandom:
    """
    The name explains itself, it randomly initializes the weights of the specified shape with values
    sampled from a uniform distribution. Uniform distribution is the one that looks like a constant function.
    """

    def initialize(self, weights_shape, fan_in, fan_out):
        return np.random.uniform(0.0, 1.0, weights_shape)


class Xavier:
    """
    Xavier initialization is suitable for sigmoid and tanh activation functions.
    By good I mean it mitigates the vanishing and exploding gradient issues.
    """

    def initialize(self, weights_shape, fan_in, fan_out):
        sigma = np.sqrt(2.0 / (fan_in + fan_out))
        return np.random.normal(0.0, sigma, weights_shape)


class He:
    """
    This is good for ReLU and mitigate its gradient problems.
    """

    def initialize(self, weights_shape, fan_in, fan_out):
        sigma = np.sqrt(2.0 / fan_in)
        """
        The first argument is the center of the distribution. It means that 
        the normal distribution is centered around zero. This means that the weights are
        equally likely to be negative or positive. Why does this matter though?
        
        If all weights start positive (or all the same sign), neurons in the same layer will all move in
        the same direction during gradient descent — they'd all 
        learn the same thing and become redundant.
        By centering at 0, roughly half the weights are positive and half negative from the start, 
        so neurons differentiate and learn different features.
        """
        return np.random.normal(0.0, sigma, weights_shape)
