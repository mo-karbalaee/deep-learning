import numpy as np
from Layers.Base import BaseLayer


class Sigmoid(BaseLayer):
    """
    Sigmoid (logistic) activation function.

        forward:   f(x) = 1 / (1 + exp(-x))     (squashes any input into (0, 1))
        backward:  f'(x) = f(x) * (1 - f(x))

    Same derivative trick as TanH: f' is expressed purely in terms of the OUTPUT
    f(x), so we cache the activation in the forward pass and reuse it in backward
    (no need to store the input). This keeps the RNN's backprop-through-time cheap.

    Why Sigmoid matters - the (0, 1) range IS the point:
        A number between 0 and 1 naturally means "what fraction to let through":
            0   -> block everything
            1   -> let everything through
            0.7 -> let 70% through
        This gives it two roles that ReLU (unbounded) cannot fill:
          - GATE (used in LSTM's forget/input/output gates): multiplying a value
            by a 0..1 gate expresses "how much to keep/forget". Multiplying by an
            unbounded ReLU output would blow the value up instead of scaling it.
          - OUTPUT (used in this Elman RNN, see RNN.forward): turns the raw output
            into a probability-like score in (0, 1).
        Both rely on the same fact: Sigmoid produces a bounded 0-to-1 number.

    Not trainable: Sigmoid has no weights, so trainable = False.
    """

    def __init__(self):
        super().__init__()
        self.trainable = False    # no learnable parameters
        self.activation = None    # caches f(x) from the forward pass for use in backward

    def forward(self, input_tensor):
        # f(x) = 1 / (1 + exp(-x)); store the result so backward can reuse it
        self.activation = 1.0 / (1.0 + np.exp(-input_tensor))
        return self.activation

    def backward(self, error_tensor):
        # chain rule: dL/dx = dL/df * f'(x), with f'(x) = f(x) * (1 - f(x))
        # (self.activation is f(x) from the forward pass)
        return error_tensor * self.activation * (1.0 - self.activation)
