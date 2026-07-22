from Layers import Conv, ReLU, Pooling, Flatten, FullyConnected, SoftMax, Initializers
from Optimization import Optimizers, Constraints
import NeuralNetwork


def build():
    """
    Build a LeNet-style CNN for MNIST (a 28x28, single-channel classification task).

    This function does not implement any new layer - it just wires together the
    pieces built earlier in the framework into the classic LeNet architecture.

    Deviations from the original LeNet (per the exercise slides):
        - input is 28x28 (not 32x32)
        - our Conv only does "same" padding, so feature maps stay larger
        - ReLU everywhere (we never implemented TanH for CNNs)
        - SoftMax + CrossEntropy at the end (instead of the original RBF units)

    Settings the task pins down:
        - Adam optimizer, learning rate 5e-4
        - L2 regularizer with weight 4e-4
    """
    # optimizer shared by all trainable layers; the network deep-copies it per layer
    optimizer = Optimizers.Adam(5e-4, 0.9, 0.999)
    optimizer.add_regularizer(Constraints.L2_Regularizer(4e-4))   # weight decay, strength 4e-4

    # He init for weights (good for ReLU), small constant 0.1 for biases
    net = NeuralNetwork.NeuralNetwork(
        optimizer,
        Initializers.He(),
        Initializers.Constant(0.1),
    )

    # We track the tensor shape (channels, H, W) as it flows through, ignoring the
    # batch dimension. Input: (1, 28, 28).

    # --- Block 1 ---
    # Conv: stride (1,1), kernel 5x5 over 1 input channel, 6 output channels.
    # "same" padding keeps H,W -> (6, 28, 28).
    net.append_layer(Conv.Conv((1, 1), (1, 5, 5), 6))
    net.append_layer(ReLU.ReLU())
    # Pool 2x2 stride 2 halves H,W -> (6, 14, 14).
    net.append_layer(Pooling.Pooling((2, 2), (2, 2)))

    # --- Block 2 ---
    # Conv: 5x5 kernel over the 6 input channels, 16 output channels.
    # "same" padding -> (16, 14, 14).
    net.append_layer(Conv.Conv((1, 1), (6, 5, 5), 16))
    net.append_layer(ReLU.ReLU())
    # Pool 2x2 stride 2 -> (16, 7, 7).
    net.append_layer(Pooling.Pooling((2, 2), (2, 2)))

    # Flatten the (16, 7, 7) volume into a single vector of length 16*7*7 = 784,
    # so the fully connected layers can consume it.
    net.append_layer(Flatten.Flatten())

    # --- Classifier head (fully connected) ---
    net.append_layer(FullyConnected.FullyConnected(16 * 7 * 7, 120))  # 784 -> 120
    net.append_layer(ReLU.ReLU())
    net.append_layer(FullyConnected.FullyConnected(120, 84))          # 120 -> 84
    net.append_layer(ReLU.ReLU())
    net.append_layer(FullyConnected.FullyConnected(84, 10))           # 84 -> 10 (one per digit)
    net.append_layer(SoftMax.SoftMax())                               # -> class probabilities

    # loss layer: SoftMax outputs probabilities, CrossEntropy scores them vs labels
    net.loss_layer = None
    from Optimization import Loss

    net.loss_layer = Loss.CrossEntropyLoss()

    return net
