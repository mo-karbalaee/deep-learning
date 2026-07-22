import numpy as np
import copy
from Layers.Base import BaseLayer
from Layers.FullyConnected import FullyConnected
from Layers.TanH import TanH
from Layers.Sigmoid import Sigmoid


class RNN(BaseLayer):
    """
    Elman Recurrent Neural Network layer (a single recurrent cell, unrolled over time).

    Big idea:
        The BATCH dimension is reinterpreted as the TIME dimension. Row t of the
        input is the input at time step t, and the rows are correlated in time
        (unlike a normal batch). We loop over time, carrying a hidden state h
        forward from one step to the next.

    Per-time-step equations (from the slides):
        h_t = tanh( [h_{t-1}, x_t, 1] . W_h )      # new hidden state
        y_t = sigmoid( [h_t, 1] . W_hy )           # output at this step

    Composite design (why this is elegant):
        Instead of managing raw weight matrices, we REUSE two FullyConnected layers
        (each already knows how to append a bias, do forward, and compute gradients):
            fc_hidden : (input_size + hidden_size) -> hidden_size    == W_h
            fc_output : hidden_size -> output_size                   == W_hy
        plus one TanH and one Sigmoid.
        "The weights of the RNN" are defined as fc_hidden's weights (the ones that
        produce the hidden state) - see the weights/gradient_weights properties.

    memorize (BPTT vs TBPTT):
        - False: hidden state resets to zeros at the start of every forward call
                 (each sequence is independent).
        - True:  the last hidden state carries over into the next forward call
                 (subsequent calls are treated as one long sequence).

    Backprop through time (the hard part):
        Because we reuse ONE fc_hidden / tanh / fc_output / sigmoid across all time
        steps, and the embedded layers only remember their LAST forward, we must
        store the per-step values in the forward pass and feed them back into the
        embedded layers when we walk backward (in reverse time). The shared weights
        mean gradients are ACCUMULATED (summed) over all time steps.
    """

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.trainable = True
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # W_h  : consumes the concatenation [hidden_state, input] -> new hidden state
        self.fc_hidden = FullyConnected(input_size + hidden_size, hidden_size)
        # W_hy : maps the hidden state -> output
        self.fc_output = FullyConnected(hidden_size, output_size)

        self.tanh = TanH()          # activation for the hidden state
        self.sigmoid = Sigmoid()    # activation for the output

        self.hidden_state = np.zeros(hidden_size)   # carried across calls if memorize=True
        self._memorize = False

        self._optimizer = None          # optimizer for fc_hidden's weights
        self._output_optimizer = None   # separate optimizer for fc_output's weights
        self._gradient_weights = None   # gradient of fc_hidden's weights (the RNN's "weights")

    @property
    def memorize(self):
        return self._memorize

    @memorize.setter
    def memorize(self, value):
        self._memorize = value

    def initialize(self, weights_initializer, bias_initializer):
        # forward initialization to both embedded FC layers
        self.fc_hidden.initialize(weights_initializer, bias_initializer)
        self.fc_output.initialize(weights_initializer, bias_initializer)

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        time_steps = input_tensor.shape[0]   # batch dim == time dim

        # first hidden state: restore the previous one (memorize) or start at zero
        if self._memorize:
            h = self.hidden_state.copy()
        else:
            h = np.zeros(self.hidden_size)

        outputs = np.zeros((time_steps, self.output_size))

        # per-time-step caches, needed for backprop through time. We store the
        # embedded layers' state at each step because a single shared layer only
        # remembers its most recent forward.
        self._hidden_fc_inputs = []     # fc_hidden.input_tensor at each step
        self._tanh_activations = []     # tanh output at each step
        self._output_fc_inputs = []     # fc_output.input_tensor at each step
        self._sigmoid_activations = []  # sigmoid output at each step

        for t in range(time_steps):
            x_t = input_tensor[t]
            # concatenate previous hidden state and current input: [h, x_t]
            # (the FC layer appends the bias "1" itself). Shape (1, hidden+input).
            concat = np.concatenate((h, x_t)).reshape(1, -1)

            # hidden pre-activation, then tanh -> new hidden state
            hidden_pre = self.fc_hidden.forward(concat)
            self._hidden_fc_inputs.append(self.fc_hidden.input_tensor)  # cache (incl. bias col)

            h = self.tanh.forward(hidden_pre)
            self._tanh_activations.append(self.tanh.activation)          # cache activation
            h = h.reshape(-1)                                            # back to 1D for next concat

            # output pre-activation, then sigmoid -> output for this step
            output_pre = self.fc_output.forward(h.reshape(1, -1))
            self._output_fc_inputs.append(self.fc_output.input_tensor)   # cache

            y = self.sigmoid.forward(output_pre)
            self._sigmoid_activations.append(self.sigmoid.activation)    # cache
            outputs[t] = y.reshape(-1)

        self.hidden_state = h   # remember final state (used next call if memorize=True)
        return outputs

    def backward(self, error_tensor):
        time_steps = error_tensor.shape[0]

        # gradient accumulators for the two shared weight matrices (summed over time)
        grad_w_hidden = np.zeros_like(self.fc_hidden.weights)
        grad_w_output = np.zeros_like(self.fc_output.weights)

        error_prev = np.zeros((time_steps, self.input_size))   # gradient w.r.t. the layer input
        grad_h_next = np.zeros((1, self.hidden_size))          # gradient flowing back from step t+1

        # walk time in REVERSE (backprop through time)
        for t in reversed(range(time_steps)):
            e = error_tensor[t].reshape(1, -1)

            # --- back through the output path: sigmoid then fc_output ---
            self.sigmoid.activation = self._sigmoid_activations[t]   # restore this step's state
            e = self.sigmoid.backward(e)

            self.fc_output.input_tensor = self._output_fc_inputs[t]  # restore this step's input
            grad_h = self.fc_output.backward(e)                      # gradient w.r.t. h from output
            grad_w_output += self.fc_output.gradient_weights         # accumulate (shared weights)

            # --- combine the two gradient paths into h ---
            # h_t affects the loss via BOTH the output y_t AND the next hidden state
            # h_{t+1}. h_t was effectively COPIED to two places, and the gradient of
            # a copy is a SUM. So add the gradient coming back from the next step.
            grad_h = grad_h + grad_h_next

            # --- back through the hidden path: tanh then fc_hidden ---
            self.tanh.activation = self._tanh_activations[t]         # restore this step's state
            grad_pre = self.tanh.backward(grad_h)

            self.fc_hidden.input_tensor = self._hidden_fc_inputs[t]  # restore this step's input
            grad_concat = self.fc_hidden.backward(grad_pre)          # gradient w.r.t. [h_{t-1}, x_t]
            grad_w_hidden += self.fc_hidden.gradient_weights         # accumulate (shared weights)

            # the concat was [h_{t-1}, x_t], so split the gradient accordingly:
            grad_h_next = grad_concat[:, : self.hidden_size]              # -> flows to previous step
            error_prev[t] = grad_concat[:, self.hidden_size :].reshape(-1)  # -> gradient for input x_t

        self._gradient_weights = grad_w_hidden   # the RNN's "weights" are fc_hidden's

        # update both weight matrices, if optimizers are attached
        if self._optimizer is not None:
            self.fc_hidden.weights = self._optimizer.calculate_update(
                self.fc_hidden.weights, grad_w_hidden
            )
        if self._output_optimizer is not None:
            self.fc_output.weights = self._output_optimizer.calculate_update(
                self.fc_output.weights, grad_w_output
            )

        return error_prev

    def calculate_regularization_loss(self):
        # RNN holds TWO weight matrices, so it gathers its own regularization loss
        # (this is the branch NeuralNetwork._regularization_loss calls for RNN/LSTM).
        reg_loss = 0.0
        if self._optimizer is not None and self._optimizer.regularizer is not None:
            reg_loss += self._optimizer.regularizer.norm(self.fc_hidden.weights)
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
        # fc_hidden and fc_output are separate weight matrices, so they need
        # separate optimizer instances (independent internal state for Adam/momentum).
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
        # "the weights of the RNN" == fc_hidden's weights (the ones computing h)
        return self.fc_hidden.weights

    @weights.setter
    def weights(self, value):
        if hasattr(self, "fc_hidden"):
            self.fc_hidden.weights = value
