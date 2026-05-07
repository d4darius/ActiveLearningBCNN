import jax
import jax.numpy as jnp

def mc_dropout_forward(apply_fn, params, rngs, x, num_samples=100, return_logits=False):
    """
    Performs stochastic forward passes using MC Dropout in JAX.
    
    Args:
        apply_fn: The model's apply function (e.g., model.apply).
        params: Model parameters.
        rngs: PRNG key for dropout.
        x: Input data.
        num_samples: Number of forward passes.
        return_logits: If True, returns logits instead of probabilities.
    """
    # Split the rng key to get independent dropout masks for each sample
    keys = jax.random.split(rngs, num_samples)
    
    def single_forward(key):
        # We pass deterministic=False to ensure dropout is applied
        logits = apply_fn({'params': params}, x, deterministic=False, rngs={'dropout': key})
        if return_logits:
            return logits
        return jax.nn.softmax(logits, axis=-1)
        
    # Vectorize the forward pass over the num_samples keys
    # This efficiently runs 'num_samples' forward passes in parallel
    mc_preds = jax.vmap(single_forward)(keys)
    return mc_preds

def compute_uncertainty(mc_probs):
    """
    Computes uncertainty metrics from Monte Carlo predictions.
    
    Args:
        mc_probs: Predictions from MC dropout of shape (num_samples, batch_size, num_classes)
        
    Returns:
        mean_probs: Mean probabilities of shape (batch_size, num_classes)
        entropy: Predictive entropy of shape (batch_size,)
        mutual_info: Mutual information (epistemic uncertainty) of shape (batch_size,)
    """
    # Predictive mean
    mean_probs = jnp.mean(mc_probs, axis=0)  # (batch_size, num_classes)
    
    # Predictive entropy (Total Uncertainty)
    # H(y|x) = - sum_c p(y=c|x) * log p(y=c|x)
    eps = 1e-10
    entropy = -jnp.sum(mean_probs * jnp.log(mean_probs + eps), axis=-1)
    
    # Expected Entropy
    # E[H(y|x, w)]
    expected_entropy = -jnp.mean(jnp.sum(mc_probs * jnp.log(mc_probs + eps), axis=-1), axis=0)
    
    # Mutual Information (Epistemic Uncertainty)
    # I(y, w|x) = H(y|x) - E[H(y|x, w)]
    mutual_info = entropy - expected_entropy
    
    return mean_probs, entropy, mutual_info
