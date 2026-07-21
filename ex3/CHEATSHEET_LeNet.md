# LeNet — Cheat Sheet

*ex3, section 1.5 (optional / bonus) · file: `Models/LeNet.py` · used by: `TrainLeNet.py`*

## What it is (one sentence)
LeNet is a classic **convolutional network for MNIST digit classification**. Our `build()` function does **not** implement a new layer — it just wires the layers we already built into the LeNet architecture and returns a ready `NeuralNetwork` object.

## Where it is used
`build()` is called by `TrainLeNet.py` (an **optional demo script, not part of the automated tests**). TrainLeNet: load MNIST → `build()` → `train(300)` → save → test → print accuracy. It proves the whole framework can train a real CNN end to end.

## Architecture + shape trace
*(channels, H, W; batch ignored)*

| Layer | Output shape | Note |
|---|---|---|
| Input | `(1, 28, 28)` | 1-channel 28×28 image |
| Conv 6 filters | `(6, 28, 28)` | 5×5 kernel, "same" padding keeps H,W |
| ReLU | | |
| Pool 2×2 | `(6, 14, 14)` | halves H,W |
| Conv 16 filters | `(16, 14, 14)` | 5×5 kernel, "same" padding |
| ReLU | | |
| Pool 2×2 | `(16, 7, 7)` | halves again |
| Flatten | `784` | **16 · 7 · 7 = 784** (this is where 784 comes from) |
| FC 784 → 120 → ReLU | | |
| FC 120 → 84 → ReLU | | |
| FC 84 → 10 | | one score per digit class |
| SoftMax | | → class probabilities |
| CrossEntropyLoss | | loss layer |

**Pattern:** pooling shrinks space (28 → 14 → 7), convs grow channels (1 → 6 → 16).

## Settings the task pins down
- **Optimizer:** Adam, learning rate `5e-4`
- **Regularizer:** L2 with weight (alpha) `4e-4`
- **Weight init:** He · **bias init:** Constant `0.1`
- ReLU everywhere, SoftMax + CrossEntropy at the end.

## Deviations from the original LeNet *(likely question!)*
- Input is **28×28**, not the original 32×32.
- Our Conv only supports **"same" padding**, so feature maps stay larger than the original (which used valid padding and shrank them).
- We only built **ReLU**, so no TanH activations like the original.
- We use **SoftMax** instead of the original's RBF/Gaussian output units.

## Save / load *(also 1.5, lives in `NeuralNetwork.py`)*
- `save(filename, net)` / `load(filename, data_layer)` use **pickle**.
- The `data_layer` is a **generator and cannot be pickled**, so `__getstate__` drops it before saving and `__setstate__` sets it back to `None` on load. That is why `load()` takes the `data_layer` again and re-attaches it.

## Likely questions → short answers
- **Where does `16*7*7` come from?** After two 2×2 poolings, 28 → 14 → 7 spatially, with 16 channels from the second conv. Flattening 16×7×7 gives 784 inputs to the first FC.
- **Why does "same" padding matter here?** It keeps H,W after each conv, so our shapes differ from the original LeNet and the input to the FC head is larger.
- **Is LeNet tested?** No — it is an optional demo run via `TrainLeNet.py`; the unit tests check individual layers, not this script.
- **Why store optimizer/initializer inside the saved network?** So a loaded network can keep training with the same settings — only the `data_layer` must be re-supplied.
