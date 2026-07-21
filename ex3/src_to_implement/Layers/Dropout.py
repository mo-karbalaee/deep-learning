import numpy as np
from Layers.Base import BaseLayer

"""
This is inverted dropout meaning, it is done during training not testing. 
The original dropout was used during testing. That's why it is called inverted. 
"""
class Dropout(BaseLayer):

    def __init__(self, probability):
        super().__init__()
        self.trainable = False
        self.probability = probability
        self.mask = None

    def forward(self, input_tensor):
        """
        It is inverted dropout hence, does not effect on the 
        testing phase. 
        """
        if self.testing_phase:
            return input_tensor
        """
        So how do we decide which neurons to drop?
        By creating a mask. A mask will sit on top of the input tensor and 
        turn off some of the nodes, but how?
        We will make a random nd array with the same shape as the input tensor
        and then make a boolean mask. 
        Then by multiplying it with the input tensor, we have canceled out 
        some of the neurons. 
        It happens on the layer level. 
        So whenever we use the dropout layer, this effect happens. 
        
        One key thing here is the division by probability. Why do we do that?
        The reason we do it is that we want the total weights that pass through 
        this layer be somewhat the same even though we have disabled some neurons. 
        Why? because we don't want to confuse the next layer by providing it with 
        less amount of weights. This disrupts the training. 
        By dividing it by the probability, it boosts the active neurons and the 
        weights that pass through them compensating for the disabled ones. 
        So for the next layer, the amount of weights coming is pretty much the same
        but only from different neurons. Nothing else. 
        """
        self.mask = (
            np.random.rand(*input_tensor.shape) < self.probability
        ) / self.probability
        return input_tensor * self.mask

    def backward(self, error_tensor):
        """
        We don't need the testing phase check here because
        during the inference we don't do back propagation dude. 
        """
        return error_tensor * self.mask
