import numpy as np
from Layers.Base import BaseLayer


class Pooling(BaseLayer):
    def __init__(self, stride_shape, pooling_shape):
        super().__init__()
        self.trainable = False
        self.stride_shape = (
            stride_shape
            if isinstance(stride_shape, (list, tuple))
            else (stride_shape, stride_shape)
        )
        self.pooling_shape = (
            pooling_shape
            if isinstance(pooling_shape, (list, tuple))
            else (pooling_shape, pooling_shape)
        )
        self.input_shape = None
        self.max_indices = None

    def forward(self, input_tensor):
        self.input_shape = input_tensor.shape
        b, c, y, x = input_tensor.shape
        sy, sx = self.stride_shape
        py, px = self.pooling_shape

        y_out = (y - py) // sy + 1
        x_out = (x - px) // sx + 1

        output_tensor = np.zeros((b, c, y_out, x_out))
        self.max_indices = {}

        for batch in range(b):
            for channel in range(c):
                for oy in range(y_out):
                    for ox in range(x_out):
                        start_y = oy * sy
                        start_x = ox * sx
                        end_y = start_y + py
                        end_x = start_x + px

                        window = input_tensor[
                            batch, channel, start_y:end_y, start_x:end_x
                        ]
                        max_idx = np.unravel_index(np.argmax(window), window.shape)

                        in_y = start_y + max_idx[0]
                        in_x = start_x + max_idx[1]

                        output_tensor[batch, channel, oy, ox] = window[max_idx]
                        self.max_indices[(batch, channel, oy, ox)] = (in_y, in_x)

        return output_tensor

    def backward(self, error_tensor):
        error_prev = np.zeros(self.input_shape)
        b, c, y_out, x_out = error_tensor.shape

        for batch in range(b):
            for channel in range(c):
                for oy in range(y_out):
                    for ox in range(x_out):
                        in_y, in_x = self.max_indices[(batch, channel, oy, ox)]
                        error_prev[batch, channel, in_y, in_x] += error_tensor[
                            batch, channel, oy, ox
                        ]

        return error_prev
