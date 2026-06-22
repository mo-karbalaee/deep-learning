import numpy as np
import copy
from scipy.signal import correlate, convolve
from Layers.Base import BaseLayer

class Conv(BaseLayer):
    def __init__(self, stride_shape, convolution_shape, num_kernels):
        super().__init__()
        self.trainable = True
        self.stride_shape = stride_shape
        self.convolution_shape = convolution_shape
        self.num_kernels = num_kernels
        
        self.is_1d = len(convolution_shape) == 2
        
        self.weights = np.random.uniform(0.0, 1.0, (num_kernels, *convolution_shape))
        self.bias = np.random.uniform(0.0, 1.0, (num_kernels,))
        
        self._optimizer_weights = None
        self._optimizer_bias = None
        self._gradient_weights = None
        self._gradient_bias = None

    @property
    def optimizer(self):
        return self._optimizer_weights

    @optimizer.setter
    def optimizer(self, opt):
        self._optimizer_weights = opt
        self._optimizer_bias = copy.deepcopy(opt)

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @property
    def gradient_bias(self):
        return self._gradient_bias

    def initialize(self, weights_initializer, bias_initializer):
        kernel_spatial_size = np.prod(self.convolution_shape[1:])
        fan_in = self.convolution_shape[0] * kernel_spatial_size
        fan_out = self.num_kernels * kernel_spatial_size
        
        self.weights = weights_initializer.initialize(
            (self.num_kernels, *self.convolution_shape), fan_in, fan_out
        )
        self.bias = bias_initializer.initialize(
            (self.num_kernels,), self.num_kernels, self.num_kernels
        )

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        
        if self.is_1d:
            batch_size, channels_in, y_in = input_tensor.shape
            x_in = 1
            input_2d = input_tensor.reshape((batch_size, channels_in, y_in, 1))
            weights_2d = self.weights.reshape((self.num_kernels, channels_in, self.convolution_shape[1], 1))
            stride_y = self.stride_shape[0] if isinstance(self.stride_shape, (list, tuple)) else self.stride_shape
            stride_x = 1
        else:
            batch_size, channels_in, y_in, x_in = input_tensor.shape
            input_2d = input_tensor
            weights_2d = self.weights
            stride_y, stride_x = self.stride_shape

        full_corr = np.zeros((batch_size, self.num_kernels, y_in, x_in))
        for b in range(batch_size):
            for k in range(self.num_kernels):
                for c in range(channels_in):
                    full_corr[b, k] += correlate(input_2d[b, c], weights_2d[k, c], mode='same')
                full_corr[b, k] += self.bias[k]

        output_2d = full_corr[:, :, ::stride_y, ::stride_x]
        
        if self.is_1d:
            return output_2d.squeeze(axis=3)
        else:
            return output_2d

    def backward(self, error_tensor):
        if self.is_1d:
            batch_size, channels_in, y_in = self.input_tensor.shape
            x_in = 1
            input_2d = self.input_tensor.reshape((batch_size, channels_in, y_in, 1))
            weights_2d = self.weights.reshape((self.num_kernels, channels_in, self.convolution_shape[1], 1))
            stride_y = self.stride_shape[0] if isinstance(self.stride_shape, (list, tuple)) else self.stride_shape
            stride_x = 1
            error_2d = error_tensor.reshape((error_tensor.shape[0], error_tensor.shape[1], error_tensor.shape[2], 1))
        else:
            batch_size, channels_in, y_in, x_in = self.input_tensor.shape
            input_2d = self.input_tensor
            weights_2d = self.weights
            stride_y, stride_x = self.stride_shape
            error_2d = error_tensor

        dilated_error = np.zeros((batch_size, self.num_kernels, y_in, x_in))
        dilated_error[:, :, ::stride_y, ::stride_x] = error_2d

        self._gradient_bias = np.sum(dilated_error, axis=(0, 2, 3))

        m, n = weights_2d.shape[2], weights_2d.shape[3]
        p_top = (m - 1) // 2
        p_bottom = m - 1 - p_top
        p_left = (n - 1) // 2
        p_right = n - 1 - p_left

        padded_input = np.pad(input_2d, ((0, 0), (0, 0), (p_top, p_bottom), (p_left, p_right)), mode='constant')

        grad_weights_2d = np.zeros_like(weights_2d)
        for k in range(self.num_kernels):
            for c in range(channels_in):
                for b in range(batch_size):
                    grad_weights_2d[k, c] += correlate(padded_input[b, c], dilated_error[b, k], mode='valid')

        if self.is_1d:
            self._gradient_weights = grad_weights_2d.reshape((self.num_kernels, *self.convolution_shape))
        else:
            self._gradient_weights = grad_weights_2d

        error_prev_2d = np.zeros((batch_size, channels_in, y_in, x_in))
        for b in range(batch_size):
            for c in range(channels_in):
                for k in range(self.num_kernels):
                    error_prev_2d[b, c] += convolve(dilated_error[b, k], weights_2d[k, c], mode='same')

        if self.is_1d:
            error_prev = error_prev_2d.squeeze(axis=3)
        else:
            error_prev = error_prev_2d

        if self._optimizer_weights:
            self.weights = self._optimizer_weights.calculate_update(self.weights, self._gradient_weights)
        if self._optimizer_bias:
            self.bias = self._optimizer_bias.calculate_update(self.bias, self._gradient_bias)

        return error_prev
