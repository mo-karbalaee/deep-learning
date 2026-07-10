import numpy as np
import copy
from Layers.Base import BaseLayer
from Layers.FullyConnected import FullyConnected
from Layers.TanH import TanH
from Layers.Sigmoid import Sigmoid


class RNN(BaseLayer):
    """
    Elman recurrent layer.

    For every time step t (the batch dimension is treated as time):
        h_t = tanh( W_h . [h_{t-1}, x_t] + b_h )
        y_t = sigmoid( W_y . h_t + b_y )

    The recurrence is implemented by reusing two FullyConnected layers (one for the
    hidden state, one for the output) plus TanH / Sigmoid activations. The FC layers
    do not own optimizers; instead the RNN accumulates their per-time-step weight
    gradients (BPTT) and performs a single update at the end of the backward pass.
    """

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.trainable = True
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # hidden FC receives the concatenation [h_{t-1}, x_t].
        self.fc_hidden = FullyConnected(input_size + hidden_size, hidden_size)
        self.fc_output = FullyConnected(hidden_size, output_size)

        self.tanh = TanH()
        self.sigmoid = Sigmoid()

        self.hidden_state = np.zeros(hidden_size)
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
        self.fc_hidden.initialize(weights_initializer, bias_initializer)
        self.fc_output.initialize(weights_initializer, bias_initializer)

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        time_steps = input_tensor.shape[0]

        # Start either from zeros or from the last iteration's hidden state.
        if self._memorize:
            h = self.hidden_state.copy()
        else:
            h = np.zeros(self.hidden_size)

        outputs = np.zeros((time_steps, self.output_size))

        # Per-time-step caches needed to restore the sub-layer states in backward.
        self._hidden_fc_inputs = []
        self._tanh_activations = []
        self._output_fc_inputs = []
        self._sigmoid_activations = []

        for t in range(time_steps):
            x_t = input_tensor[t]
            concat = np.concatenate((h, x_t)).reshape(1, -1)

            hidden_pre = self.fc_hidden.forward(concat)
            self._hidden_fc_inputs.append(self.fc_hidden.input_tensor)

            h = self.tanh.forward(hidden_pre)
            self._tanh_activations.append(self.tanh.activation)
            h = h.reshape(-1)

            output_pre = self.fc_output.forward(h.reshape(1, -1))
            self._output_fc_inputs.append(self.fc_output.input_tensor)

            y = self.sigmoid.forward(output_pre)
            self._sigmoid_activations.append(self.sigmoid.activation)
            outputs[t] = y.reshape(-1)

        # Remember the last hidden state for a possible next (memorized) sequence.
        self.hidden_state = h
        return outputs

    def backward(self, error_tensor):
        time_steps = error_tensor.shape[0]

        grad_w_hidden = np.zeros_like(self.fc_hidden.weights)
        grad_w_output = np.zeros_like(self.fc_output.weights)

        error_prev = np.zeros((time_steps, self.input_size))
        grad_h_next = np.zeros((1, self.hidden_size))

        for t in reversed(range(time_steps)):
            e = error_tensor[t].reshape(1, -1)

            # Back through the output sigmoid and output FC.
            self.sigmoid.activation = self._sigmoid_activations[t]
            e = self.sigmoid.backward(e)

            self.fc_output.input_tensor = self._output_fc_inputs[t]
            grad_h = self.fc_output.backward(e)
            grad_w_output += self.fc_output.gradient_weights

            # Add the gradient flowing back from the future time step.
            grad_h = grad_h + grad_h_next

            # Back through the hidden tanh and hidden FC.
            self.tanh.activation = self._tanh_activations[t]
            grad_pre = self.tanh.backward(grad_h)

            self.fc_hidden.input_tensor = self._hidden_fc_inputs[t]
            grad_concat = self.fc_hidden.backward(grad_pre)
            grad_w_hidden += self.fc_hidden.gradient_weights

            # Split the concatenated gradient back into hidden and input parts.
            grad_h_next = grad_concat[:, :self.hidden_size]
            error_prev[t] = grad_concat[:, self.hidden_size:].reshape(-1)

        self._gradient_weights = grad_w_hidden

        # Single update using the accumulated (summed) gradients.
        if self._optimizer is not None:
            self.fc_hidden.weights = self._optimizer.calculate_update(self.fc_hidden.weights, grad_w_hidden)
        if self._output_optimizer is not None:
            self.fc_output.weights = self._output_optimizer.calculate_update(self.fc_output.weights, grad_w_output)

        return error_prev

    def calculate_regularization_loss(self):
        reg_loss = 0.0
        if self._optimizer is not None and self._optimizer.regularizer is not None:
            reg_loss += self._optimizer.regularizer.norm(self.fc_hidden.weights)
        if self._output_optimizer is not None and self._output_optimizer.regularizer is not None:
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

    # The weights of the RNN are the weights of the hidden-state FC layer.
    @property
    def weights(self):
        return self.fc_hidden.weights

    @weights.setter
    def weights(self, value):
        # Guard against BaseLayer.__init__ assigning weights = None before the
        # inner FC layers exist.
        if hasattr(self, 'fc_hidden'):
            self.fc_hidden.weights = value
