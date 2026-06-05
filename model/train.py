import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from libs.loss import compute_metrics, cross_entropy_loss
from functools import partial

class TrainState(train_state.TrainState):
    """Custom train state if we need additional info, else use standard."""
    pass

def create_train_state(rng, model, learning_rate, weight_decay, input_shape):
    """Creates initial `TrainState`."""
    # Initialize the model weights
    params = model.init(rng, jnp.ones(input_shape), deterministic=True)['params']
    
    # Use AdamW to handle weight decay properly (corresponds to prior variance)
    tx = optax.adamw(learning_rate, weight_decay=weight_decay)
    
    return TrainState.create(apply_fn=model.apply, params=params, tx=tx)

@partial(jax.jit, static_argnames=['is_deterministic'])
def train_step(state, batch, dropout_rng, is_deterministic=False):
    """Trains for a single step."""
    def loss_fn(params):
        logits = state.apply_fn(
            {'params': params}, 
            batch['image'], 
            deterministic=is_deterministic, 
            rngs={'dropout': dropout_rng}
        )
        loss = cross_entropy_loss(logits, batch['label'])
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    
    metrics = compute_metrics(logits, batch['label'])
    return state, metrics

@jax.jit
def eval_step(state, batch):
    """Evaluates for a single step with dropout disabled (standard inference)."""
    logits = state.apply_fn(
        {'params': state.params}, 
        batch['image'], 
        deterministic=True
    )
    return compute_metrics(logits, batch['label'])

def train_epoch(state, train_ds, batch_size, rng, is_deterministic=False):
    """Train for a single epoch. Assumes train_ds is a dict with 'image' and 'label'."""
    train_ds_size = len(train_ds['image'])
    # Clamp batch size to dataset size — critical for small initial sets
    effective_batch_size = min(batch_size, train_ds_size)
    
    perms = jax.random.permutation(rng, train_ds_size)

    # Build batch index ranges, merging any small remainder into the last
    # full batch to avoid noisy gradient updates from tiny batches
    # (e.g. n=130, bs=128 → one batch of 130 instead of [128, 2])
    n_full = train_ds_size // effective_batch_size
    remainder = train_ds_size % effective_batch_size

    batch_slices = []
    for i in range(n_full):
        start = i * effective_batch_size
        end = start + effective_batch_size
        # Extend the last full batch to absorb the remainder
        if i == n_full - 1 and remainder > 0:
            end = train_ds_size
        batch_slices.append((start, end))

    # Edge case: dataset smaller than batch_size (handled by min above,
    # so n_full >= 1 always holds, but be safe)
    if not batch_slices:
        batch_slices.append((0, train_ds_size))
    
    batch_metrics = []
    # Process ALL data including the remainder batch
    for start, end in batch_slices:
        perm = perms[start:end]
        batch = {k: v[perm] for k, v in train_ds.items()}
        rng, dropout_rng = jax.random.split(rng)
        state, metrics = train_step(state, batch, dropout_rng, is_deterministic=is_deterministic)
        batch_metrics.append(metrics)
        
    # compute mean of metrics across batches
    batch_metrics_np = jax.device_get(batch_metrics)
    epoch_metrics_np = {
        k: jnp.mean(jnp.array([m[k] for m in batch_metrics_np]))
        for k in batch_metrics_np[0]
    }
    return state, epoch_metrics_np, rng

def evaluate(state, test_ds, batch_size):
    """Evaluate on the test set."""
    test_ds_size = len(test_ds['image'])
    
    batch_metrics = []
    # Process ALL test data including the remainder batch
    for start in range(0, test_ds_size, batch_size):
        batch = {k: v[start:start + batch_size] for k, v in test_ds.items()}
        metrics = eval_step(state, batch)
        batch_metrics.append(metrics)
        
    batch_metrics_np = jax.device_get(batch_metrics)
    epoch_metrics_np = {
        k: jnp.mean(jnp.array([m[k] for m in batch_metrics_np]))
        for k in batch_metrics_np[0]
    }
    return epoch_metrics_np
