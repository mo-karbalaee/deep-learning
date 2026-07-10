class BaseLayer:
    def __init__(self):
        # Whether the layer holds trainable parameters (weights/bias).
        self.trainable = False
        # Some layers (Dropout, BatchNorm) behave differently during training
        # and testing. The NeuralNetwork toggles this flag through its 'phase'
        # property before training/testing. Default is training (False).
        self.testing_phase = False
        self.weights = None
        self.input_tensor = None
