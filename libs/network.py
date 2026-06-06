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

class VGG16(nn.Module):
    """
    VGG16 architecture with MC Dropout for Bayesian Active Learning.
    Matches torchvision.models.vgg16 architecture.
    """
    num_classes: int = 2
    dropout_prob: float = 0.5
    @nn.compact
    def __call__(self, x, deterministic: bool = False):
        # Block 1
        x = nn.Conv(features=64, kernel_size=(3, 3), padding='SAME', name='features_0')(x)
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(3, 3), padding='SAME', name='features_2')(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        # Block 2
        x = nn.Conv(features=128, kernel_size=(3, 3), padding='SAME', name='features_5')(x)
        x = nn.relu(x)
        x = nn.Conv(features=128, kernel_size=(3, 3), padding='SAME', name='features_7')(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        # Block 3
        x = nn.Conv(features=256, kernel_size=(3, 3), padding='SAME', name='features_10')(x)
        x = nn.relu(x)
        x = nn.Conv(features=256, kernel_size=(3, 3), padding='SAME', name='features_12')(x)
        x = nn.relu(x)
        x = nn.Conv(features=256, kernel_size=(3, 3), padding='SAME', name='features_14')(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        # Block 4
        x = nn.Conv(features=512, kernel_size=(3, 3), padding='SAME', name='features_17')(x)
        x = nn.relu(x)
        x = nn.Conv(features=512, kernel_size=(3, 3), padding='SAME', name='features_19')(x)
        x = nn.relu(x)
        x = nn.Conv(features=512, kernel_size=(3, 3), padding='SAME', name='features_21')(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        # Block 5
        x = nn.Conv(features=512, kernel_size=(3, 3), padding='SAME', name='features_24')(x)
        x = nn.relu(x)
        x = nn.Conv(features=512, kernel_size=(3, 3), padding='SAME', name='features_26')(x)
        x = nn.relu(x)
        x = nn.Conv(features=512, kernel_size=(3, 3), padding='SAME', name='features_28')(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        # Adaptive Avg Pool equivalent for 224x224 input -> 7x7 output
        # For 224x224, 5 max pools of size 2 reduce it to 7x7. 
        # So we don't strictly need adaptive pooling if input is 224x224.
        
        # Flatten
        x = x.reshape((x.shape[0], -1))
        # Classifier Block
        x = nn.Dense(features=4096, name='classifier_0')(x)
        x = nn.relu(x)
        x = nn.Dropout(rate=self.dropout_prob, deterministic=deterministic)(x)
        x = nn.Dense(features=4096, name='classifier_3')(x)
        x = nn.relu(x)
        x = nn.Dropout(rate=self.dropout_prob, deterministic=deterministic)(x)
        x = nn.Dense(features=self.num_classes, name='classifier_6')(x)
        return x

def load_pretrained_vgg16(rng, input_shape=(1, 224, 224, 3), num_classes=2, dropout_prob=0.5):
    """
    Initializes VGG16 in Flax and loads pre-trained weights from torchvision.
    Returns the loaded params.
    """
    import torchvision.models as models
    import jax
    import jax.numpy as jnp
    # Initialize Flax model
    model = VGG16(num_classes=num_classes, dropout_prob=dropout_prob)
    init_rngs = {'params': rng}
    params = model.init(init_rngs, jnp.ones(input_shape), deterministic=True)['params']
    # Unfreeze Flax params (flax returns FrozenDict)
    from flax.core import unfreeze, freeze
    params = unfreeze(params)
    # Load PyTorch pre-trained model
    pt_model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    pt_state = pt_model.state_dict()
    def load_conv2d(flax_layer_name, pt_layer_idx):
        # PyTorch: (out_channels, in_channels, kH, kW)
        # Flax: (kH, kW, in_channels, out_channels)
        weight = pt_state[f'features.{pt_layer_idx}.weight'].numpy()
        bias = pt_state[f'features.{pt_layer_idx}.bias'].numpy()
        
        weight = np.transpose(weight, (2, 3, 1, 0))
        params[flax_layer_name]['kernel'] = jnp.array(weight)
        params[flax_layer_name]['bias'] = jnp.array(bias)
    def load_linear(flax_layer_name, pt_layer_idx):
        # PyTorch: (out_features, in_features)
        # Flax: (in_features, out_features)
        weight = pt_state[f'classifier.{pt_layer_idx}.weight'].numpy()
        bias = pt_state[f'classifier.{pt_layer_idx}.bias'].numpy()
        
        # The first linear layer connects from conv features.
        # PyTorch flatten is (B, C, H, W) -> (B, C*H*W)
        # Flax flatten is (B, H, W, C) -> (B, H*W*C)
        # We need to reshape the PyTorch weights to match Flax's flattening order for the first Dense layer.
        if pt_layer_idx == 0:
            # weight shape: (4096, 25088) -> (4096, 512, 7, 7)
            weight = weight.reshape(4096, 512, 7, 7)
            # Transpose to match (4096, 7, 7, 512)
            weight = np.transpose(weight, (0, 2, 3, 1))
            # Flatten back
            weight = weight.reshape(4096, 25088)
            
        weight = np.transpose(weight, (1, 0))
        params[flax_layer_name]['kernel'] = jnp.array(weight)
        params[flax_layer_name]['bias'] = jnp.array(bias)
    import numpy as np
    
    # Map layers
    conv_mapping = {
        'features_0': 0, 'features_2': 2,
        'features_5': 5, 'features_7': 7,
        'features_10': 10, 'features_12': 12, 'features_14': 14,
        'features_17': 17, 'features_19': 19, 'features_21': 21,
        'features_24': 24, 'features_26': 26, 'features_28': 28
    }
    for flax_name, pt_idx in conv_mapping.items():
        load_conv2d(flax_name, pt_idx)
    # We only load the first two linear layers since the last one is now 2 classes instead of 1000
    linear_mapping = {
        'classifier_0': 0, 'classifier_3': 3
    }
    for flax_name, pt_idx in linear_mapping.items():
        load_linear(flax_name, pt_idx)
    return freeze(params), model