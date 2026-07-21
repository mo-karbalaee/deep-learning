import numpy as np


class L2_Regularizer:

    def __init__(self, alpha):
        self.alpha = alpha

    """
    This is the part that allows L2 to make bigger weights much smaller. 
    """
    def calculate_gradient(self, weights):
        return self.alpha * weights
    
    """
    Forward path. 
    """
    def norm(self, weights):
        return self.alpha * np.sum(np.square(weights))


class L1_Regularizer:

    def __init__(self, alpha):
        self.alpha = alpha

    def calculate_gradient(self, weights):
        """
        The calculate gradient functions are only used in the backward path. 
        You see that in the L1 regularization, the gradient does not care
        about the magnitude of the weights but only their sign. 
        np.sign is element-wise, meaning it will calculate the sign of
        each element and return the array of the same shape. 
        """
        return self.alpha * np.sign(weights)

    def norm(self, weights):
        return self.alpha * np.sum(np.abs(weights))
