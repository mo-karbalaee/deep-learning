import numpy as np
import copy
from Layers.Base import BaseLayer
from Layers import Helpers


class BatchNormalization(BaseLayer):
    """
    Batch Normalization layer.

    Idea:
        1) Normalize each feature to zero-mean / unit-variance over the batch:
               X̃ = (X - μ_B) / sqrt(σ²_B + ε)
        2) Let the network re-scale and re-shift with two LEARNABLE parameters:
               Ŷ = γ · X̃ + β
           (γ = self.weights, β = self.bias)
           
        We want them to be learnable because the pure normalization might 
        destroy useful information. For example, maybe this distribution and mean
        and everything is not good for that specific feature. There has to be
        some wiggle room here. These learnable parameters are the wiggle room. To 
        normalize accordingly.    

    Why:
        Keeps activations in a stable range -> smoother loss landscape, faster
        training, and a mild regularizing effect (batch statistics are noisy).
        γ and β let the network undo the normalization if that is useful, so no
        representational power is lost.

    Two behaviors (controlled by self.testing_phase, set network-wide via the
    `phase` property):
        - training: normalize with the CURRENT batch's stats, and keep a moving
                    average of those stats on the side.
        - testing:  normalize with the MOVING AVERAGE collected during training
                    (test predictions must be deterministic and independent of
                    which other samples share the batch).

    Works for both:
        - vector input:  shape (B, channels)             -> ndim == 2
        - image input:   shape (B, channels, H, W)       -> ndim == 4
      The image case is handled by reshaping to the vector case (see reformat),
      running the exact same code, then reshaping back.
    """

    def __init__(self, channels):
        super().__init__()
        self.trainable = True          # has learnable params (γ, β) -> the network attaches an optimizer
        self.channels = channels       # number of features (vector) / channels (image)
        self.epsilon = 1e-11           # added under the sqrt to avoid /0; task requires < 1e-10
        self.alpha = 0.8               # decay for the moving-average of mean/var
        """
        This alpha is used in the exponential moving average that we used for the 
        std and mean during training. The higher the alpha value the higher 
        the effect of the new values into the average. 
        """
        self.initialize(None, None)    # set γ=1, β=0 (identity at the start of training)

        # --- values cached in forward, needed in backward ---
        self._mean = None              # batch mean μ_B of the last forward
        self._var = None               # batch variance σ²_B of the last forward
        self._normalized = None        # X̃ (normalized input) of the last forward
        self._input_vec = None         # the (possibly reshaped) input fed to normalization

        # --- moving-average statistics used at test time ---
        self._moving_mean = None       # μ̃ : running estimate of the training-set mean
        self._moving_var = None        # σ̃²: running estimate of the training-set variance

        self._image_shape = None       # remembers original 4D shape so reformat can reverse it

        # --- optimizers / gradients (γ and β are optimized separately) ---
        self._optimizer = None         # optimizer for γ (self.weights)
        self._bias_optimizer = None    # optimizer for β (self.bias) - own copy (own internal state!)
        self._gradient_weights = None  # ∂L/∂γ
        self._gradient_bias = None     # ∂L/∂β

    def initialize(self, weights_initializer, bias_initializer):
        """
        γ and β are ALWAYS initialized to ones/zeros (the passed initializers are
        intentionally ignored). This makes the layer start as the identity:
            Ŷ = 1·X̃ + 0 = X̃
        so BatchNorm does not disrupt the network at the beginning of training.
        γ and β only drift away from 1/0 if the optimizer finds it useful.
        """
        self.weights = np.ones(self.channels)   # γ
        self.bias = np.zeros(self.channels)      # β

    def reformat(self, tensor):
        """
        Bridge between the image case (4D) and the vector case (2D).

        We want ONE μ,σ per channel, aggregating over the batch AND all spatial
        positions. Trick: reshape so the channel axis becomes the "feature" axis
        and (batch · height · width) becomes the "batch" axis. Then the ordinary
        vector-case code (np.mean(x, axis=0)) computes exactly the right thing.

        Called with a 4D tensor  -> returns 2D  (image  -> vector)
        Called with a 2D tensor  -> returns 4D  (vector -> image), using the
                                     shape stored during the 4D->2D call.
        """
        if tensor.ndim == 4:
            # image -> vector
            self._image_shape = tensor.shape          # remember (B, C, H, W) to reverse later
            b, c, h, w = tensor.shape
            tensor = tensor.reshape(b, c, h * w)       # B × C × (H·W)
            tensor = np.transpose(tensor, (0, 2, 1))   # B × (H·W) × C   (move channels to the end)
            return tensor.reshape(b * h * w, c)        # (B·H·W) × C   -> C is now the feature axis
        else:
            # vector -> image (exact reverse of the branch above)
            b, c, h, w = self._image_shape
            tensor = tensor.reshape(b, h * w, c)
            tensor = np.transpose(tensor, (0, 2, 1))
            return tensor.reshape(b, c, h, w)

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        is_conv = input_tensor.ndim == 4               # 4D => image case

        # image -> 2D so the rest of the method is shape-agnostic
        x = self.reformat(input_tensor) if is_conv else input_tensor
        self._input_vec = x                            # cache for backward

        if self.testing_phase:
            # TEST: use the moving-average stats collected during training
            mean = self._moving_mean
            var = self._moving_var
        else:
            # TRAIN: use THIS batch's statistics
            mean = np.mean(x, axis=0)                  # μ_B, one value per feature
            var = np.var(x, axis=0)                    # σ²_B, one value per feature

            # update the moving average on the side (used later, at test time)
            if self._moving_mean is None:
                # first batch ever -> seed the estimate with it
                self._moving_mean = mean
                self._moving_var = var
            else:
                # exponential moving average: keep α of the old, mix in (1-α) of the new
                self._moving_mean = (
                    self.alpha * self._moving_mean + (1.0 - self.alpha) * mean
                )
                self._moving_var = (
                    self.alpha * self._moving_var + (1.0 - self.alpha) * var
                )

            # cache the batch stats for the backward pass
            self._mean = mean
            self._var = var

        # normalize, then apply the learnable scale/shift:  Ŷ = γ·X̃ + β
        self._normalized = (x - mean) / np.sqrt(var + self.epsilon)   # X̃
        out = self.weights * self._normalized + self.bias             # Ŷ

        # 2D -> image on the way out (if we came in as an image)
        return self.reformat(out) if is_conv else out

    def backward(self, error_tensor):
        is_conv = error_tensor.ndim == 4
        # image -> 2D so the gradient math matches the forward
        err = self.reformat(error_tensor) if is_conv else error_tensor

        # --- gradients w.r.t. the learnable parameters (computed by hand) ---
        self._gradient_weights = np.sum(err * self._normalized, axis=0)  # ∂L/∂γ = Σ (error · X̃)
        self._gradient_bias = np.sum(err, axis=0)                        # ∂L/∂β = Σ error

        # --- gradient w.r.t. the input (the messy formula, provided for us) ---
        # Note: compute_bn_gradients ONLY returns the input gradient, not γ/β grads.
        grad_input = Helpers.compute_bn_gradients(
            err, self._input_vec, self.weights, self._mean, self._var, self.epsilon
        )

        # update γ and β, but ONLY if optimizers are attached (task requirement)
        if self._optimizer is not None:
            self.weights = self._optimizer.calculate_update(
                self.weights, self._gradient_weights
            )
        if self._bias_optimizer is not None:
            self.bias = self._bias_optimizer.calculate_update(
                self.bias, self._gradient_bias
            )

        # 2D -> image on the way out
        return self.reformat(grad_input) if is_conv else grad_input

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, opt):
        # γ and β are separate tensors, so they need SEPARATE optimizer instances.
        # Stateful optimizers (Adam/momentum) hold state shaped like their parameter,
        # so we deepcopy to give β its own independent state.
        self._optimizer = opt
        self._bias_optimizer = copy.deepcopy(opt)

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @property
    def gradient_bias(self):
        return self._gradient_bias
