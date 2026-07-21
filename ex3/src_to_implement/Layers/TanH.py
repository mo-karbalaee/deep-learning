import numpy as np
from Layers.Base import BaseLayer


class TanH(BaseLayer):
    """
    Hyperbolic tangent activation function.

        forward:   f(x) = tanh(x)          (squashes any input into the range (-1, 1))
        backward:  f'(x) = 1 - f(x)^2

    Key trick (why we store the OUTPUT, not the input):
        The derivative can be written purely in terms of the OUTPUT f(x), not the
        original input x. So instead of caching the input, we cache the activation
        (the forward result) and reuse it in the backward pass. This is the same
        idea used for Sigmoid, and it is what makes the RNN's backprop-through-time
        cheap: at every time step we just need the stored activation.

    Why TanH (and Sigmoid) here instead of ReLU?
        In an RNN the hidden state feeds back into itself every time step, so the
        same signal recirculates over and over. That changes what a good
        activation is:
          - TanH is BOUNDED to (-1, 1): the recirculating state can never blow up,
            so the recurrence stays stable. ReLU is unbounded on the positive side,
            so feeding it back into itself repeatedly can make the state EXPLODE
            (much worse in recurrence than in a feed-forward CNN, where each
            activation is used only once).
          - TanH is zero-centered, so the state can go negative (useful memory);
            ReLU only outputs >= 0.
          - Sigmoid's (0, 1) range is exactly what gates/probabilities need
            (0 = block, 1 = pass) - the RNN output and all LSTM gates use it.
        ReLU's unboundedness is a feature in CNNs but a liability in recurrence,
        which is why the classic Elman RNN / LSTM are defined with TanH + Sigmoid.

    Not trainable: TanH has no weights, so trainable = False.
    """

    def __init__(self):
        super().__init__()
        self.trainable = False    # no learnable parameters
        self.activation = None    # caches f(x) from the forward pass for use in backward

    def forward(self, input_tensor):
        # f(x) = tanh(x); store the result so backward can reuse it
        self.activation = np.tanh(input_tensor)
        return self.activation

    def backward(self, error_tensor):
        # chain rule: dL/dx = dL/df * f'(x), with f'(x) = 1 - f(x)^2
        # (self.activation is f(x) from the forward pass)
        return error_tensor * (1.0 - np.square(self.activation))
