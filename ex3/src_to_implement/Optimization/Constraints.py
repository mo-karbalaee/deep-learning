import numpy as np

"""
Regularizers add a penalty on the magnitude of the weights to the loss. They serve
two purposes and therefore expose two methods:

- calculate_gradient(weights): returns the (sub-)gradient of the penalty w.r.t. the
  weights. The optimizer uses it to shrink the weights during the update step.
- norm(weights): returns the scalar penalty value (already scaled by alpha). The
  NeuralNetwork adds this to the data loss to report the norm-enhanced loss.

Both are scaled by alpha, the regularization weight passed to the constructor.
"""


class L2_Regularizer:
    """
    L2 regularization (weight decay). Penalizes large weights, encouraging small
    weights overall. The gradient of alpha/2 * ||w||^2 would be alpha * w; here the
    penalty reported by norm() is alpha * sum(w^2) and the gradient is alpha * w.
    """

    def __init__(self, alpha):
        self.alpha = alpha

    def calculate_gradient(self, weights):
        # Gradient of the L2 penalty: proportional to the weights themselves.
        return self.alpha * weights

    def norm(self, weights):
        # Squared L2 norm scaled by alpha.
        return self.alpha * np.sum(np.square(weights))


class L1_Regularizer:
    """
    L1 regularization. Penalizes the absolute value of the weights, encouraging
    sparsity (many weights driven exactly to zero). The (sub-)gradient of |w| is
    the element-wise sign of w.
    """

    def __init__(self, alpha):
        self.alpha = alpha

    def calculate_gradient(self, weights):
        # Sub-gradient of the L1 penalty: sign of each weight.
        return self.alpha * np.sign(weights)

    def norm(self, weights):
        # Sum of absolute values scaled by alpha.
        return self.alpha * np.sum(np.abs(weights))
