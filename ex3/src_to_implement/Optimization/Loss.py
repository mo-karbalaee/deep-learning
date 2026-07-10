import numpy as np


class CrossEntropyLoss:
    def __init__(self):
        self.prediction_tensor = None

    def forward(self, prediction_tensor, label_tensor):
        self.prediction_tensor = prediction_tensor
        if label_tensor is None:
            return prediction_tensor

        if label_tensor.ndim == 1:
            num_classes = prediction_tensor.shape[1]
            one_hot_labels = np.zeros((label_tensor.shape[0], num_classes))
            one_hot_labels[np.arange(label_tensor.shape[0]), label_tensor] = 1
            label_tensor = one_hot_labels

        epsilon = np.finfo(float).eps
        prediction_tensor_clipped = np.clip(prediction_tensor, epsilon, 1 - epsilon)
        loss_per_sample = -np.sum(
            label_tensor * np.log(prediction_tensor_clipped), axis=1
        )
        return np.sum(loss_per_sample)

    def backward(self, label_tensor):
        if label_tensor is None:
            return np.zeros_like(self.prediction_tensor)

        if label_tensor.ndim == 1:
            num_classes = self.prediction_tensor.shape[1]
            one_hot_labels = np.zeros((label_tensor.shape[0], num_classes))
            one_hot_labels[np.arange(label_tensor.shape[0]), label_tensor] = 1
            label_tensor = one_hot_labels

        epsilon = np.finfo(float).eps
        prediction_tensor_clipped = np.clip(
            self.prediction_tensor, epsilon, 1 - epsilon
        )
        return -label_tensor / prediction_tensor_clipped
