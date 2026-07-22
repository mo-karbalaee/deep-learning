# LSTM — Cheat Sheet

*ex3, section 2.1 (optional / bonus) · file: `Layers/LSTM.py`*

## What it is (one sentence)
A more powerful recurrent cell than the Elman RNN: it adds a **cell state `c`** that lets gradients flow across many time steps without vanishing, so it remembers over longer spans.

## Why it beats the plain RNN *(the whole point)*
Elman RNNs suffer from the **vanishing-gradient problem**: gradients shrink as they flow back through many tanh steps, so long-range memory is lost. The LSTM's cell state `c` is updated with mostly **add and multiply**:
```
c_t = f * c_{t-1} + i * c_tilde
```
This "conveyor belt" lets gradients pass with far less decay.

## Two states carried across time *(vs. the RNN's one)*
- `h` : hidden state (also the cell's output each step)
- `c` : cell state (long-term memory)

## Four gates — all from ONE FC layer *(efficiency trick, likely question)*
A single FullyConnected of output size `4*hidden_size` produces all four pre-activations; we slice its output into quarters:

| Gate | Activation | Job | Range |
|---|---|---|---|
| `f` (forget) | sigmoid | how much **old** cell state to keep | 0..1 |
| `i` (input) | sigmoid | how much of the new candidate to add | 0..1 |
| `c̃` (candidate) | tanh | the candidate values to maybe add | −1..1 |
| `o` (output) | sigmoid | how much of the cell to expose as `h` | 0..1 |

So there are only **two** weight matrices total:
- `fc_gates` : `(input_size + hidden_size) → 4*hidden_size`
- `fc_output` : `hidden_size → output_size`

**"The weights of the LSTM" = `fc_gates`' weights.**

## The update equations *(per time step)*
```
concat   = [h_{t-1}, x_t]
f,i,c~,o = gates from fc_gates(concat)   (sigmoid, sigmoid, tanh, sigmoid)
c_t = f * c_{t-1} + i * c_tilde          forget some old, write some new
h_t = o * tanh(c_t)                      gated view of the cell
y_t = sigmoid( fc_output(h_t) )          output
```
The three sigmoids **are** the gates — "fraction to let through" (0 = block, 1 = pass). This is exactly what ReLU cannot express.

## `memorize` (BPTT vs TBPTT)
Same idea as the RNN, but **both `h` and `c`** carry over between forward calls when `memorize` is True (reset to zeros otherwise).

## Backward = Backprop Through Time
Walk time in **reverse**. Now **two** gradients flow back between steps: `grad_h_next` **and** `grad_c_next`. Two "gradient-is-a-sum" copy points:
```
grad_h = grad_h + grad_h_next    (h used by output AND next step)
grad_c = (from tanh(c) this step) + grad_c_next   (c used here AND next c)
```
Then plain chain rule / product rule:
- `h_new = o * tanh(c)`: `grad_o = grad_h*tanh_c` ; `grad_tanh_c = grad_h*o`
- `c_new = f*c_prev + i*c_tilde` (product rule → each factor's grad is the **other** factor times incoming grad):
  - `grad_f = grad_c*c_prev` ; `grad_c_prev = grad_c*f`
  - `grad_i = grad_c*c_tilde` ; `grad_c_tilde = grad_c*i`
- Each gate then goes back through its own activation derivative (sigmoid: `g*(1-g)`; tanh: `1 - t^2`), the four pieces are concatenated in the **same order** they were sliced, and fed into `fc_gates.backward`.

Split `grad_concat` (`[h_{t-1}, x_t]`):
- `[:, :hidden]` → `grad_h_next` (previous step's `h`)
- `[:, hidden:]` → error for input `x_t`

Carry `grad_c_next = grad_c_prev` to the previous step. Weights are shared across time → accumulate with `+=`.

## Regularization
Like the RNN: two weight matrices, so LSTM has its own `calculate_regularization_loss()` summing norms of `fc_gates` and `fc_output`.

## Likely questions → short answers
- **Why does LSTM solve vanishing gradients?** The cell state `c` is updated with `+` and `*`, so gradients flow back along it with little decay, unlike the RNN's repeated tanh squashing.
- **Why one big FC of size `4*hidden`?** All four gates read the same concat `[h, x]`; one FC computes all four pre-activations at once, then we slice — fewer objects, one matrix.
- **Why are gates sigmoids?** Their `(0, 1)` output means "fraction to let through" — forget/keep/expose.
- **What extra thing must backward carry vs the RNN?** A cell-state gradient `grad_c_next`, in addition to `grad_h_next`.
