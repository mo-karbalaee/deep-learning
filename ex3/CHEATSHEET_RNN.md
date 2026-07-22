# Elman RNN — Cheat Sheet

*ex3, section 2.1 · file: `Layers/RNN.py`*

## What it is (one sentence)
A recurrent layer that processes a **sequence one time step at a time**, carrying a hidden state `h` forward so each output depends on the past.

## The key reinterpretation
The **batch dimension is treated as the time dimension**. Row `t` of the input is the input at time step `t`, and the rows are **correlated in time** (unlike a normal batch, where samples are independent).

## The equations *(per time step)*
```
h_t = tanh( [h_{t-1}, x_t, 1] . W_h )     new hidden state
y_t = sigmoid( [h_t, 1] . W_hy )          output at this step
```
`[...]` = concatenation; the trailing `1` is the bias the FC layer adds.

## Composite design *(why it is elegant — likely question)*
We do **not** manage raw weight matrices. We **reuse two FullyConnected layers** (each already knows bias, forward, and gradients) plus TanH and Sigmoid:
- `fc_hidden` : `(input_size + hidden_size) → hidden_size`  == `W_h`
- `fc_output` : `hidden_size → output_size`  == `W_hy`

**"The weights of the RNN" are defined as `fc_hidden`'s weights.**

## `memorize` (BPTT vs TBPTT)
- **False:** hidden state resets to zeros at the start of every forward call (each sequence independent).
- **True:** the last hidden state carries over into the next forward call (subsequent calls treated as one long sequence).

Implemented by storing `self.hidden_state` at the end of `forward` and restoring it at the start if `memorize` is True.

## Forward — what happens *(loop over time steps)*
1. `h` = previous hidden state (memorize) or zeros.
2. For each `t`: `concat = [h, x_t]`; feed through `fc_hidden → tanh → new h`; feed `h` through `fc_output → sigmoid → y_t`.
3. **Store per-step values** (fc inputs, tanh & sigmoid activations) in lists, because a single shared layer only remembers its **last** forward and BPTT needs every step.
4. Save final `h` for the next call.

## Backward = Backprop Through Time *(THE hard part)*
Walk time in **reverse**. The single most important line:
```
grad_h = grad_h + grad_h_next
```
**Why it is a sum:** `h_t` influences the loss via **two paths** — the output `y_t` **and** the next hidden state `h_{t+1}`. `h_t` was effectively **copied** to two places, and the gradient of a copy is a **sum**. So we add the gradient coming back from the future step.

Other key points:
- **Restore** each embedded layer's cached state before calling its backward (e.g. set `self.tanh.activation` = stored value for that step).
- Weights are **shared** across time, so gradients are **accumulated** with `+=` (`grad_w_hidden += ...`, `grad_w_output += ...`).
- The concat was `[h_{t-1}, x_t]`, so split `grad_concat`:
  - `[:, :hidden_size]` → `grad_h_next` (flows to previous step)
  - `[:, hidden_size:]` → error for input `x_t` (returned to previous layer)

## Regularization
RNN holds **two** weight matrices, so it has its own `calculate_regularization_loss()` summing the norm of both `fc_hidden` and `fc_output` weights. `NeuralNetwork` calls this method for RNN/LSTM instead of the generic single-weight path.

## Activations — why TanH and Sigmoid (not ReLU)
- **TanH** bounds the hidden state to `(-1, 1)` so the recirculating state stays stable; ReLU is unbounded and can **explode** across time steps.
- **Sigmoid**'s `(0, 1)` output is a natural probability for `y_t`.

## Likely questions → short answers
- **Why store activations per time step?** One shared TanH/Sigmoid/FC remembers only its last forward; BPTT revisits every step, so we cache each step's values and replay them.
- **Why is `grad_h` a sum?** `h_t` is used twice (output + next hidden state); gradient of a copy sums.
- **What are the RNN's weights?** `fc_hidden`'s weights (the ones producing the hidden state).
- **Why two separate optimizers?** `fc_hidden` and `fc_output` are different matrices; stateful optimizers (Adam/momentum) need independent internal state, so we deepcopy.
