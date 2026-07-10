class BaseLayer:
    def __init__(self):
        self.trainable = False
        self.testing_phase = False
        self.weights = None
        self.input_tensor = None
