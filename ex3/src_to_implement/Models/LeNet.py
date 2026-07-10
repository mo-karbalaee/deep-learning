from Layers import Conv, ReLU, Pooling, Flatten, FullyConnected, SoftMax, Initializers
from Optimization import Optimizers, Constraints
import NeuralNetwork


def build():
    """
    Build a LeNet-style convolutional network for MNIST (1x28x28 images).

    Uses 'same'-padded convolutions (the original LeNet used valid convolutions;
    the task explicitly tells us to ignore that difference), ReLU activations, max
    pooling and a SoftMax output. The optimizer is ADAM with an L2 regularizer.
    """
    optimizer = Optimizers.Adam(5e-4, 0.9, 0.999)
    optimizer.add_regularizer(Constraints.L2_Regularizer(4e-4))

    net = NeuralNetwork.NeuralNetwork(
        optimizer,
        Initializers.He(),
        Initializers.Constant(0.1),
    )

    # 1x28x28 -> 6x28x28
    net.append_layer(Conv.Conv((1, 1), (1, 5, 5), 6))
    net.append_layer(ReLU.ReLU())
    # 6x28x28 -> 6x14x14
    net.append_layer(Pooling.Pooling((2, 2), (2, 2)))

    # 6x14x14 -> 16x14x14
    net.append_layer(Conv.Conv((1, 1), (6, 5, 5), 16))
    net.append_layer(ReLU.ReLU())
    # 16x14x14 -> 16x7x7
    net.append_layer(Pooling.Pooling((2, 2), (2, 2)))

    net.append_layer(Flatten.Flatten())

    net.append_layer(FullyConnected.FullyConnected(16 * 7 * 7, 120))
    net.append_layer(ReLU.ReLU())
    net.append_layer(FullyConnected.FullyConnected(120, 84))
    net.append_layer(ReLU.ReLU())
    net.append_layer(FullyConnected.FullyConnected(84, 10))
    net.append_layer(SoftMax.SoftMax())

    net.loss_layer = None  # set by the training script together with the data layer
    from Optimization import Loss
    net.loss_layer = Loss.CrossEntropyLoss()

    return net
