import jax
import jax.numpy as jnp
import optax
import numpyro.distributions as dist

def cross_entropy_loss(logits, labels):
    """
    Standard Cross Entropy Loss for classification in JAX/Optax.
    Expects labels to be integer class indices.
    
    Note on Bayesian formulation:
    In the context of Monte Carlo Dropout, minimizing the cross-entropy loss 
    along with L2 regularization (weight decay) is mathematically equivalent to 
    minimizing the Kullback-Leibler (KL) divergence between the approximate 
    posterior and the true posterior.
    """
    # One-hot encode the labels
    labels_onehot = jax.nn.one_hot(labels, num_classes=logits.shape[-1])
    
    # Calculate softmax cross entropy
    loss = optax.softmax_cross_entropy(logits=logits, labels=labels_onehot)
    
    return jnp.mean(loss)

def numpyro_nll_loss(logits, labels):
    """
    Alternative implementation using NumPyro distributions.
    This calculates the negative log-likelihood under a Categorical distribution.
    """
    categorical = dist.Categorical(logits=logits)
    return -jnp.mean(categorical.log_prob(labels))

def compute_metrics(logits, labels):
    loss = cross_entropy_loss(logits, labels)
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    return {'loss': loss, 'accuracy': accuracy}
