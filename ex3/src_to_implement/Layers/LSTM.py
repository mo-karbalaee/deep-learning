import numpy as np
import copy
from Layers.Base import BaseLayer
from Layers.FullyConnected import FullyConnected


class LSTM(BaseLayer):

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.trainable = True
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.fc_gates = FullyConnected(input_size + hidden_size, 4 * hidden_size)
        self.fc_output = FullyConnected(hidden_size, output_size)

        self.hidden_state = np.zeros(hidden_size)
        self.cell_state = np.zeros(hidden_size)
        self._memorize = False

        self._optimizer = None
        self._output_optimizer = None
        self._gradient_weights = None

    @property
    def memorize(self):
        return self._memorize

    @memorize.setter
    def memorize(self, value):
        self._memorize = value

    def initialize(self, weights_initializer, bias_initializer):
        self.fc_gates.initialize(weights_initializer, bias_initializer)
        self.fc_output.initialize(weights_initializer, bias_initializer)

    @staticmethod
    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        time_steps = input_tensor.shape[0]
        h_dim = self.hidden_size

        if self._memorize:
            h = self.hidden_state.copy()
            c = self.cell_state.copy()
        else:
            h = np.zeros(h_dim)
            c = np.zeros(h_dim)

        outputs = np.zeros((time_steps, self.output_size))

        self._cache = []

        for t in range(time_steps):
            x_t = input_tensor[t]
            concat = np.concatenate((h, x_t)).reshape(1, -1)

            gates_pre = self.fc_gates.forward(concat)
            gate_input = self.fc_gates.input_tensor

            f_pre = gates_pre[:, 0 * h_dim : 1 * h_dim]
            i_pre = gates_pre[:, 1 * h_dim : 2 * h_dim]
            c_pre = gates_pre[:, 2 * h_dim : 3 * h_dim]
            o_pre = gates_pre[:, 3 * h_dim : 4 * h_dim]

            f = self._sigmoid(f_pre)
            i = self._sigmoid(i_pre)
            c_tilde = np.tanh(c_pre)
            o = self._sigmoid(o_pre)

            c_prev = c.reshape(1, -1)
            c_new = f * c_prev + i * c_tilde
            tanh_c = np.tanh(c_new)
            h_new = o * tanh_c

            output_pre = self.fc_output.forward(h_new)
            output_fc_input = self.fc_output.input_tensor
            y = self._sigmoid(output_pre)

            outputs[t] = y.reshape(-1)

            self._cache.append(
                {
                    "gate_input": gate_input,
                    "output_fc_input": output_fc_input,
                    "f": f,
                    "i": i,
                    "c_tilde": c_tilde,
                    "o": o,
                    "c_prev": c_prev,
                    "c_new": c_new,
                    "tanh_c": tanh_c,
                    "y": y,
                }
            )

            h = h_new.reshape(-1)
            c = c_new.reshape(-1)

        self.hidden_state = h
        self.cell_state = c
        return outputs

    def backward(self, error_tensor):
        time_steps = error_tensor.shape[0]
        h_dim = self.hidden_size

        grad_w_gates = np.zeros_like(self.fc_gates.weights)
        grad_w_output = np.zeros_like(self.fc_output.weights)

        error_prev = np.zeros((time_steps, self.input_size))
        grad_h_next = np.zeros((1, h_dim))
        grad_c_next = np.zeros((1, h_dim))

        for t in reversed(range(time_steps)):
            cache = self._cache[t]
            e = error_tensor[t].reshape(1, -1)

            y = cache["y"]
            e = e * y * (1.0 - y)
            self.fc_output.input_tensor = cache["output_fc_input"]
            grad_h = self.fc_output.backward(e)
            grad_w_output += self.fc_output.gradient_weights

            grad_h = grad_h + grad_h_next

            o = cache["o"]
            tanh_c = cache["tanh_c"]
            grad_o = grad_h * tanh_c
            grad_tanh_c = grad_h * o

            grad_c = grad_tanh_c * (1.0 - tanh_c**2) + grad_c_next

            f = cache["f"]
            i = cache["i"]
            c_tilde = cache["c_tilde"]
            c_prev = cache["c_prev"]

            grad_f = grad_c * c_prev
            grad_c_prev = grad_c * f
            grad_i = grad_c * c_tilde
            grad_c_tilde = grad_c * i

            grad_f_pre = grad_f * f * (1.0 - f)
            grad_i_pre = grad_i * i * (1.0 - i)
            grad_c_pre = grad_c_tilde * (1.0 - c_tilde**2)
            grad_o_pre = grad_o * o * (1.0 - o)

            grad_gates_pre = np.concatenate(
                (grad_f_pre, grad_i_pre, grad_c_pre, grad_o_pre), axis=1
            )

            self.fc_gates.input_tensor = cache["gate_input"]
            grad_concat = self.fc_gates.backward(grad_gates_pre)
            grad_w_gates += self.fc_gates.gradient_weights

            grad_h_next = grad_concat[:, :h_dim]
            error_prev[t] = grad_concat[:, h_dim:].reshape(-1)
            grad_c_next = grad_c_prev

        self._gradient_weights = grad_w_gates

        if self._optimizer is not None:
            self.fc_gates.weights = self._optimizer.calculate_update(
                self.fc_gates.weights, grad_w_gates
            )
        if self._output_optimizer is not None:
            self.fc_output.weights = self._output_optimizer.calculate_update(
                self.fc_output.weights, grad_w_output
            )

        return error_prev

    def calculate_regularization_loss(self):
        reg_loss = 0.0
        if self._optimizer is not None and self._optimizer.regularizer is not None:
            reg_loss += self._optimizer.regularizer.norm(self.fc_gates.weights)
        if (
            self._output_optimizer is not None
            and self._output_optimizer.regularizer is not None
        ):
            reg_loss += self._output_optimizer.regularizer.norm(self.fc_output.weights)
        return reg_loss

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, opt):
        self._optimizer = opt
        self._output_optimizer = copy.deepcopy(opt)

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, value):
        self._gradient_weights = value

    @property
    def weights(self):
        return self.fc_gates.weights

    @weights.setter
    def weights(self, value):
        if hasattr(self, "fc_gates"):
            self.fc_gates.weights = value
