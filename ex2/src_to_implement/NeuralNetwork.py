import copy

class NeuralNetwork:
    def __init__(self, optimizer, weights_initializer, bias_initializer):
        self.optimizer = optimizer
        self.weights_initializer = weights_initializer
        self.bias_initializer = bias_initializer
        self.loss = []
        self.layers = []
        self.data_layer = None
        self.loss_layer = None

    def append_layer(self, layer):
        if hasattr(layer, 'trainable') and layer.trainable:
            layer.optimizer = copy.deepcopy(self.optimizer)
            if hasattr(layer, 'initialize'):
                layer.initialize(self.weights_initializer, self.bias_initializer)
        self.layers.append(layer)

    def forward(self):
        self.input_tensor, self.label_tensor = self.data_layer.next()
        out = self.input_tensor
        for layer in self.layers:
            out = layer.forward(out)
        return self.loss_layer.forward(out, self.label_tensor)

    def backward(self):
        error_tensor = self.loss_layer.backward(self.label_tensor)
        for layer in reversed(self.layers):
            error_tensor = layer.backward(error_tensor)

    def train(self, iterations):
        for _ in range(iterations):
            loss = self.forward()
            self.loss.append(loss)
            self.backward()

    def test(self, input_tensor):
        out = input_tensor
        for layer in self.layers:
            out = layer.forward(out)
        return self.loss_layer.forward(out, None)
