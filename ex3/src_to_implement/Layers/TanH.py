import numpy as np
from Layers.Base import BaseLayer


class TanH(BaseLayer):
    """
    Hyperbolic tangent activation. It squashes its input into the range (-1, 1).

    We store the *activation* (the output) rather than the input, because the
    derivative of tanh can be written purely in terms of its output:
        d/dx tanh(x) = 1 - tanh(x)^2
    This is the dynamic-programming trick mentioned in the task.
    """

    def __init__(self):
        super().__init__()
        self.trainable = False
        self.activation = None

    def forward(self, input_tensor):
        self.activation = np.tanh(input_tensor)
        return self.activation

    def backward(self, error_tensor):
        return error_tensor * (1.0 - np.square(self.activation))
