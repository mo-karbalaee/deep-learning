import numpy as np
import copy
from Layers.Base import BaseLayer
from Layers.FullyConnected import FullyConnected


class LSTM(BaseLayer):
    """
    Long Short-Term Memory layer (optional). Same skeleton as the Elman RNN, but a
    more powerful cell that fixes the vanishing-gradient problem via a CELL STATE.

    Two states carried across time (vs. the RNN's one):
        h : hidden state (also the output of the cell each step)
        c : cell state   - the "long-term memory conveyor belt". Gradients flow
            along c with mostly + and * operations, so they vanish far less than
            in a plain RNN -> the network can remember over longer spans.

    Four gates, all computed from the same concatenation [h_{t-1}, x_t]:
        f (forget) : sigmoid - how much of the OLD cell state to keep    (0..1)
        i (input)  : sigmoid - how much of the new candidate to write in (0..1)
        c_tilde    : tanh    - the candidate values to possibly add      (-1..1)
        o (output) : sigmoid - how much of the cell state to expose as h (0..1)

    Cell / hidden update:
        c_t = f * c_{t-1} + i * c_tilde       # forget some old, add some new
        h_t = o * tanh(c_t)                   # expose a gated view of the cell

    Efficiency trick (matches the slide):
        All 4 gate pre-activations come from ONE FullyConnected layer with output
        size 4*hidden_size; we just slice its output into the four pieces. So there
        are only two weight matrices total:
            fc_gates  : (input_size + hidden_size) -> 4*hidden_size
            fc_output : hidden_size -> output_size
        The "weights of the LSTM" are defined as fc_gates' weights.

    Backprop through time:
        As in the RNN, we reuse the embedded layers across all time steps, so we
        cache everything per step in self._cache and replay it in reverse. Two
        gradients now flow back between steps: grad_h_next AND grad_c_next.
    """

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.trainable = True
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # one FC produces all four gate pre-activations at once (4 * hidden_size)
        self.fc_gates = FullyConnected(input_size + hidden_size, 4 * hidden_size)
        self.fc_output = FullyConnected(hidden_size, output_size)

        self.hidden_state = np.zeros(hidden_size)   # h carried across calls if memorize=True
        self.cell_state = np.zeros(hidden_size)      # c carried across calls if memorize=True
        self._memorize = False

        self._optimizer = None          # optimizer for fc_gates
        self._output_optimizer = None   # separate optimizer for fc_output
        self._gradient_weights = None   # gradient of fc_gates' weights

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
        time_steps = input_tensor.shape[0]   # batch dim == time dim
        h_dim = self.hidden_size

        # restore h AND c from the previous call (memorize) or start both at zero
        if self._memorize:
            h = self.hidden_state.copy()
            c = self.cell_state.copy()
        else:
            h = np.zeros(h_dim)
            c = np.zeros(h_dim)

        outputs = np.zeros((time_steps, self.output_size))

        self._cache = []   # per-step values needed for backprop through time

        for t in range(time_steps):
            x_t = input_tensor[t]
            concat = np.concatenate((h, x_t)).reshape(1, -1)   # [h_{t-1}, x_t]

            # one FC -> 4*hidden pre-activations, then slice into the four gates
            gates_pre = self.fc_gates.forward(concat)
            gate_input = self.fc_gates.input_tensor

            f_pre = gates_pre[:, 0 * h_dim : 1 * h_dim]
            i_pre = gates_pre[:, 1 * h_dim : 2 * h_dim]
            c_pre = gates_pre[:, 2 * h_dim : 3 * h_dim]
            o_pre = gates_pre[:, 3 * h_dim : 4 * h_dim]

            f = self._sigmoid(f_pre)        # forget gate (0..1)
            i = self._sigmoid(i_pre)        # input gate  (0..1)
            c_tilde = np.tanh(c_pre)         # candidate   (-1..1)
            o = self._sigmoid(o_pre)         # output gate (0..1)

            # update the cell state, then produce the hidden state / output
            c_prev = c.reshape(1, -1)
            c_new = f * c_prev + i * c_tilde     # forget some old + add some new
            tanh_c = np.tanh(c_new)
            h_new = o * tanh_c                   # gated view of the cell

            output_pre = self.fc_output.forward(h_new)
            output_fc_input = self.fc_output.input_tensor
            y = self._sigmoid(output_pre)

            outputs[t] = y.reshape(-1)

            # cache everything the backward pass will need for this step
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

            h = h_new.reshape(-1)   # carry to next step
            c = c_new.reshape(-1)

        self.hidden_state = h   # remember final states (used next call if memorize=True)
        self.cell_state = c
        return outputs

    def backward(self, error_tensor):
        time_steps = error_tensor.shape[0]
        h_dim = self.hidden_size

        # gradient accumulators for the two shared weight matrices (summed over time)
        grad_w_gates = np.zeros_like(self.fc_gates.weights)
        grad_w_output = np.zeros_like(self.fc_output.weights)

        error_prev = np.zeros((time_steps, self.input_size))
        grad_h_next = np.zeros((1, h_dim))   # gradient of h flowing back from step t+1
        grad_c_next = np.zeros((1, h_dim))   # gradient of c flowing back from step t+1

        for t in reversed(range(time_steps)):
            cache = self._cache[t]
            e = error_tensor[t].reshape(1, -1)

            # --- back through output sigmoid + fc_output ---
            y = cache["y"]
            e = e * y * (1.0 - y)                        # sigmoid'(output) = y(1-y)
            self.fc_output.input_tensor = cache["output_fc_input"]
            grad_h = self.fc_output.backward(e)
            grad_w_output += self.fc_output.gradient_weights

            # h_t goes to BOTH the output and the next step -> gradients sum
            grad_h = grad_h + grad_h_next

            # --- h_new = o * tanh(c_new): split gradient to o and to tanh(c) ---
            o = cache["o"]
            tanh_c = cache["tanh_c"]
            grad_o = grad_h * tanh_c                     # dh/do
            grad_tanh_c = grad_h * o                     # dh/d tanh(c)

            # c_t affects loss via tanh(c) here AND via the next step's cell state
            grad_c = grad_tanh_c * (1.0 - tanh_c**2) + grad_c_next   # tanh' + future c grad

            # --- c_new = f*c_prev + i*c_tilde: split gradient (product rule) ---
            f = cache["f"]
            i = cache["i"]
            c_tilde = cache["c_tilde"]
            c_prev = cache["c_prev"]

            grad_f = grad_c * c_prev        # d c_new / d f
            grad_c_prev = grad_c * f        # d c_new / d c_prev  -> flows to previous step
            grad_i = grad_c * c_tilde       # d c_new / d i
            grad_c_tilde = grad_c * i        # d c_new / d c_tilde

            # --- back through each gate's own activation to its pre-activation ---
            grad_f_pre = grad_f * f * (1.0 - f)              # sigmoid'
            grad_i_pre = grad_i * i * (1.0 - i)              # sigmoid'
            grad_c_pre = grad_c_tilde * (1.0 - c_tilde**2)   # tanh'
            grad_o_pre = grad_o * o * (1.0 - o)              # sigmoid'

            # re-assemble the four pre-activation gradients in the SAME order we sliced
            grad_gates_pre = np.concatenate(
                (grad_f_pre, grad_i_pre, grad_c_pre, grad_o_pre), axis=1
            )

            # --- back through the single gate FC ---
            self.fc_gates.input_tensor = cache["gate_input"]
            grad_concat = self.fc_gates.backward(grad_gates_pre)
            grad_w_gates += self.fc_gates.gradient_weights

            # concat was [h_{t-1}, x_t]; split gradient and set up the carries for t-1
            grad_h_next = grad_concat[:, :h_dim]                 # -> previous step's h
            error_prev[t] = grad_concat[:, h_dim:].reshape(-1)   # -> gradient for input x_t
            grad_c_next = grad_c_prev                            # -> previous step's c

        self._gradient_weights = grad_w_gates   # the LSTM's "weights" are fc_gates'

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
        # like the RNN: two weight matrices, so gather their regularization norms here
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
        # fc_gates and fc_output need independent optimizer instances (own state)
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
        # "the weights of the LSTM" == fc_gates' weights
        return self.fc_gates.weights

    @weights.setter
    def weights(self, value):
        if hasattr(self, "fc_gates"):
            self.fc_gates.weights = value
