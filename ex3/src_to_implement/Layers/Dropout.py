import numpy as np
from Layers.Base import BaseLayer


class Dropout(BaseLayer):
    """
    Inverted dropout. During training a random fraction of the activations is set to
    zero and the surviving activations are scaled up by 1/probability, so that the
    expected value of the output matches the input. Because of this scaling nothing
    has to be changed during testing: the layer simply passes the input through.

    probability: the fraction of units to *keep* (often called p).
    """

    def __init__(self, probability):
        super().__init__()
        self.trainable = False
        self.probability = probability
        self.mask = None

    def forward(self, input_tensor):
        if self.testing_phase:
            # At test time inverted dropout is the identity.
            return input_tensor
        # Keep a unit with probability p, then scale kept units by 1/p.
        self.mask = (np.random.rand(*input_tensor.shape) < self.probability) / self.probability
        return input_tensor * self.mask

    def backward(self, error_tensor):
        # Route the gradient only through the units that were kept, with the same
        # 1/p scaling applied in the forward pass.
        return error_tensor * self.mask
