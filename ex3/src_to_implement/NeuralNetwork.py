import copy
import pickle


class NeuralNetwork:
    def __init__(self, optimizer, weights_initializer, bias_initializer):
        self.optimizer = optimizer
        self.weights_initializer = weights_initializer
        self.bias_initializer = bias_initializer
        self.loss = []
        self.layers = []
        self.data_layer = None
        self.loss_layer = None
        self._phase = False

    @property
    def phase(self):
        return self._phase

    @phase.setter
    def phase(self, value):
        self._phase = value
        for layer in self.layers:
            layer.testing_phase = value

    def append_layer(self, layer):
        if hasattr(layer, "trainable") and layer.trainable:
            layer.optimizer = copy.deepcopy(self.optimizer)
            if hasattr(layer, "initialize"):
                layer.initialize(self.weights_initializer, self.bias_initializer)
        self.layers.append(layer)

    def _regularization_loss(self):
        reg_loss = 0.0
        for layer in self.layers:
            if hasattr(layer, "calculate_regularization_loss"):
                reg_loss += layer.calculate_regularization_loss()
            elif getattr(layer, "trainable", False):
                optimizer = getattr(layer, "optimizer", None)
                if (
                    optimizer is not None
                    and getattr(optimizer, "regularizer", None) is not None
                    and layer.weights is not None
                ):
                    reg_loss += optimizer.regularizer.norm(layer.weights)
        return reg_loss

    def forward(self):
        self.input_tensor, self.label_tensor = self.data_layer.next()
        out = self.input_tensor
        for layer in self.layers:
            out = layer.forward(out)
        data_loss = self.loss_layer.forward(out, self.label_tensor)
        return data_loss + self._regularization_loss()

    def backward(self):
        error_tensor = self.loss_layer.backward(self.label_tensor)
        for layer in reversed(self.layers):
            error_tensor = layer.backward(error_tensor)

    def train(self, iterations):
        self.phase = False
        for _ in range(iterations):
            loss = self.forward()
            self.loss.append(loss)
            self.backward()

    def test(self, input_tensor):
        self.phase = True
        out = input_tensor
        for layer in self.layers:
            out = layer.forward(out)
        return self.loss_layer.forward(out, None)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["data_layer"] = None
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        self.data_layer = None


def save(filename, net):
    data_layer = net.data_layer
    net.data_layer = None
    try:
        with open(filename, "wb") as f:
            pickle.dump(net, f)
    finally:
        net.data_layer = data_layer


def load(filename, data_layer):
    with open(filename, "rb") as f:
        net = pickle.load(f)
    net.data_layer = data_layer
    return net
