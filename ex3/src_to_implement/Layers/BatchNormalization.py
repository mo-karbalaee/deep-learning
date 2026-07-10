import numpy as np
import copy
from Layers.Base import BaseLayer
from Layers import Helpers


class BatchNormalization(BaseLayer):
    """
    Batch normalization with trainable scale (gamma / weights) and shift
    (beta / bias) per channel.

    It works both on vector-like input (batch, channels) and on image-like input
    (batch, channels, height, width). For the image case the normalization statistics
    are computed per channel over the batch *and* the spatial dimensions; the
    'reformat' method converts between the two representations so the same vector-case
    math can be reused.

    During training the batch statistics are used and a moving average of mean and
    variance is maintained. During testing the moving statistics are used instead.
    """

    def __init__(self, channels):
        super().__init__()
        self.trainable = True
        self.channels = channels
        # epsilon must be smaller than 1e-10 (see task / Helpers.compute_bn_gradients).
        self.epsilon = 1e-11
        # moving average decay for the running mean/variance.
        self.alpha = 0.8

        self.initialize(None, None)

        # Statistics of the current forward pass (needed in the backward pass).
        self._mean = None
        self._var = None
        self._normalized = None
        self._input_vec = None

        # Running estimates used during the testing phase.
        self._moving_mean = None
        self._moving_var = None

        # Shape of the last image-like input, used by reformat to go vector->image.
        self._image_shape = None

        self._optimizer = None
        self._bias_optimizer = None
        self._gradient_weights = None
        self._gradient_bias = None

    def initialize(self, weights_initializer, bias_initializer):
        # gamma (weights) starts at ones and beta (bias) at zeros so that batch norm
        # is the identity transform at the beginning of training. Any provided
        # initializer is intentionally ignored.
        self.weights = np.ones(self.channels)
        self.bias = np.zeros(self.channels)

    def reformat(self, tensor):
        if tensor.ndim == 4:
            # image -> vector: (B, C, H, W) -> (B*H*W, C)
            self._image_shape = tensor.shape
            b, c, h, w = tensor.shape
            tensor = tensor.reshape(b, c, h * w)
            tensor = np.transpose(tensor, (0, 2, 1))
            return tensor.reshape(b * h * w, c)
        else:
            # vector -> image: (B*H*W, C) -> (B, C, H, W) using the stored image shape.
            b, c, h, w = self._image_shape
            tensor = tensor.reshape(b, h * w, c)
            tensor = np.transpose(tensor, (0, 2, 1))
            return tensor.reshape(b, c, h, w)

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        is_conv = input_tensor.ndim == 4

        # Work in the vector representation regardless of the input shape.
        x = self.reformat(input_tensor) if is_conv else input_tensor
        self._input_vec = x

        if self.testing_phase:
            mean = self._moving_mean
            var = self._moving_var
        else:
            mean = np.mean(x, axis=0)
            var = np.var(x, axis=0)

            # Initialize / update the moving statistics.
            if self._moving_mean is None:
                self._moving_mean = mean
                self._moving_var = var
            else:
                self._moving_mean = self.alpha * self._moving_mean + (1.0 - self.alpha) * mean
                self._moving_var = self.alpha * self._moving_var + (1.0 - self.alpha) * var

            self._mean = mean
            self._var = var

        self._normalized = (x - mean) / np.sqrt(var + self.epsilon)
        out = self.weights * self._normalized + self.bias

        return self.reformat(out) if is_conv else out

    def backward(self, error_tensor):
        is_conv = error_tensor.ndim == 4
        err = self.reformat(error_tensor) if is_conv else error_tensor

        # Gradients w.r.t. the trainable parameters (summed over the batch).
        self._gradient_weights = np.sum(err * self._normalized, axis=0)
        self._gradient_bias = np.sum(err, axis=0)

        # Gradient w.r.t. the input, delegated to the provided helper.
        grad_input = Helpers.compute_bn_gradients(
            err, self._input_vec, self.weights, self._mean, self._var, self.epsilon
        )

        # Only update the parameters if optimizers have been assigned.
        if self._optimizer is not None:
            self.weights = self._optimizer.calculate_update(self.weights, self._gradient_weights)
        if self._bias_optimizer is not None:
            self.bias = self._bias_optimizer.calculate_update(self.bias, self._gradient_bias)

        return self.reformat(grad_input) if is_conv else grad_input

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, opt):
        self._optimizer = opt
        # A separate copy keeps the bias-optimizer's internal state independent.
        self._bias_optimizer = copy.deepcopy(opt)

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @property
    def gradient_bias(self):
        return self._gradient_bias
