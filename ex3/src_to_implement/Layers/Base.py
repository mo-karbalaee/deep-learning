class BaseLayer:
    def __init__(self):
        self.trainable = False
        """
        Useful for dropout and batch norm because they behave 
        differently. 
        During the training dropout works but during testing it should be 
        completely turned off. 
        For batch norm, during training the normalization happens by 
        using the mean and std of the current batch, but during testing, 
        it happens by the moving average of mean and std that we calculated
        during training. Why? because we might want to pass only one item
        during inference. then mean and std will be zero and it collapses. 
        It's worth reminding you that a moving average means we don't store
        the items then calculate the average but we keep updating it over time.
        directly!! Also normalization means subtracting mean and dividing by 
        std. 
        """
        self.testing_phase = False
        self.weights = None
        self.input_tensor = None
