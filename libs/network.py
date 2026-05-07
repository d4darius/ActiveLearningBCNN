import jax.numpy as jnp
from flax import linen as nn

class BayesianCNN(nn.Module):
    """
    Bayesian Convolutional Neural Network (BCNN) with Dropout for Monte Carlo sampling.
    """
    num_classes: int = 10
    dropout_prob: float = 0.5

    @nn.compact
    def __call__(self, x, deterministic: bool = False):
        # Input shape: (batch_size, H, W, C)
        x = nn.Conv(features=32, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Dropout(rate=self.dropout_prob / 2, deterministic=deterministic)(x)
        
        x = nn.Conv(features=64, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Dropout(rate=self.dropout_prob / 2, deterministic=deterministic)(x)
        
        # Flatten
        x = x.reshape((x.shape[0], -1))
        
        x = nn.Dense(features=128)(x)
        x = nn.relu(x)
        x = nn.Dropout(rate=self.dropout_prob, deterministic=deterministic)(x)
        
        x = nn.Dense(features=self.num_classes)(x)
        return x
