import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from libs.loss import compute_metrics, cross_entropy_loss

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

@jax.jit
def train_step(state, batch, dropout_rng):
    """Trains for a single step."""
    def loss_fn(params):
        logits = state.apply_fn(
            {'params': params}, 
            batch['image'], 
            deterministic=False, 
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

def train_epoch(state, train_ds, batch_size, rng):
    """Train for a single epoch. Assumes train_ds is a dict with 'image' and 'label'."""
    train_ds_size = len(train_ds['image'])
    steps_per_epoch = train_ds_size // batch_size
    
    perms = jax.random.permutation(rng, train_ds_size)
    perms = perms[:steps_per_epoch * batch_size]  # skip incomplete batch
    perms = perms.reshape((steps_per_epoch, batch_size))
    
    batch_metrics = []
    for perm in perms:
        batch = {k: v[perm] for k, v in train_ds.items()}
        rng, dropout_rng = jax.random.split(rng)
        state, metrics = train_step(state, batch, dropout_rng)
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
    steps = test_ds_size // batch_size
    
    batch_metrics = []
    for i in range(steps):
        batch = {k: v[i * batch_size:(i + 1) * batch_size] for k, v in test_ds.items()}
        metrics = eval_step(state, batch)
        batch_metrics.append(metrics)
        
    batch_metrics_np = jax.device_get(batch_metrics)
    epoch_metrics_np = {
        k: jnp.mean(jnp.array([m[k] for m in batch_metrics_np]))
        for k in batch_metrics_np[0]
    }
    return epoch_metrics_np
