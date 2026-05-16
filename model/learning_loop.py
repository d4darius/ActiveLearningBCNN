import jax
import jax.numpy as jnp
import numpy as np
from functools import partial

from model.train import create_train_state, train_epoch, evaluate
from libs.utils import mc_dropout_forward, compute_uncertainty


# ---------------------------------------------------------------------------
# Acquisition functions
# Each takes (entropy, mutual_info) of shape (pool_size,)
# and returns a score of shape (pool_size,) — higher = more informative
# ---------------------------------------------------------------------------

def acquisition_bald(entropy, mutual_info):
    """BALD: maximise mutual information (epistemic uncertainty)."""
    return mutual_info

def acquisition_max_entropy(entropy, mutual_info):
    """Max Entropy: maximise predictive entropy (total uncertainty)."""
    return entropy

def acquisition_variation_ratios(mean_probs):
    """
    Variation Ratios: 1 - max predicted probability.
    Needs mean_probs of shape (pool_size, num_classes).
    """
    return 1.0 - jnp.max(mean_probs, axis=-1)

def acquisition_mean_std(mc_probs):
    """
    Mean STD: average std across classes over MC samples.
    mc_probs shape: (num_samples, pool_size, num_classes)
    """
    std_per_class = jnp.std(mc_probs, axis=0)          # (pool_size, num_classes)
    return jnp.mean(std_per_class, axis=-1)             # (pool_size,)

def acquisition_random(entropy, rng):
    """Random baseline: uniform scores."""
    return jax.random.uniform(rng, shape=entropy.shape)

ACQUISITION_FUNCTIONS = ['bald', 'max_entropy', 'variation_ratios', 'mean_std', 'random']


# ---------------------------------------------------------------------------
# Pool / train set helpers
# ---------------------------------------------------------------------------

def append_to_train(train_set, pool, indices):
    """
    Moves selected pool points into the training set.

    Args:
        train_set : dict {'image': jnp array, 'label': jnp array}
        pool      : dict {'image': jnp array, 'label': jnp array}
        indices   : 1-D numpy array of pool indices to move

    Returns:
        Updated train_set dict with the new points appended.
    """
    return {
        'image': jnp.concatenate([train_set['image'], pool['image'][indices]], axis=0),
        'label': jnp.concatenate([train_set['label'], pool['label'][indices]], axis=0),
    }

def remove_from_pool(pool, indices):
    """
    Removes selected points from the pool after acquisition.

    Args:
        pool    : dict {'image': jnp array, 'label': jnp array}
        indices : 1-D numpy array of pool indices to remove

    Returns:
        Updated pool dict with selected points removed.
    """
    mask = np.ones(len(pool['image']), dtype=bool)
    mask[indices] = False
    return {
        'image': pool['image'][mask],
        'label': pool['label'][mask],
    }

def get_balanced_initial_set(dataset, n_per_class=2, num_classes=10, rng=None):
    """
    Samples a small balanced initial training set from the dataset.
    The paper starts with 20 points (2 per class for MNIST).

    Args:
        dataset     : dict {'image': ..., 'label': ...}
        n_per_class : number of labelled examples per class
        num_classes : number of classes
        rng         : numpy random Generator (optional)

    Returns:
        init_train : small balanced dict
        pool       : remaining dict
    """
    if rng is None:
        rng = np.random.default_rng(42)

    labels = np.array(dataset['label'])
    selected = []
    for c in range(num_classes):
        class_idx = np.where(labels == c)[0]
        chosen = rng.choice(class_idx, size=n_per_class, replace=False)
        selected.extend(chosen.tolist())

    selected = np.array(selected)
    mask = np.ones(len(labels), dtype=bool)
    mask[selected] = False

    init_train = {
        'image': dataset['image'][selected],
        'label': dataset['label'][selected],
    }
    pool = {
        'image': dataset['image'][mask],
        'label': dataset['label'][mask],
    }
    return init_train, pool


# ---------------------------------------------------------------------------
# Core scoring function
# ---------------------------------------------------------------------------

def score_pool(apply_fn, params, rng, pool_images,
               acquisition_name, num_mc_samples=10, batch_size=512):
    """
    Runs MC dropout over the pool in batches and returns acquisition scores.

    Batching is important: running mc_dropout_forward on 59K images at once
    will OOM on most GPUs/TPUs.

    Args:
        apply_fn         : model.apply
        params           : current model parameters
        rng              : JAX PRNG key
        pool_images      : jnp array of shape (pool_size, H, W, C)
        acquisition_name : one of ACQUISITION_FUNCTIONS
        num_mc_samples   : number of stochastic forward passes (paper uses 10)
        batch_size       : images per batch during scoring

    Returns:
        scores : numpy array of shape (pool_size,)
    """
    n = len(pool_images)
    all_scores = []

    for start in range(0, n, batch_size):
        batch = pool_images[start: start + batch_size]
        rng, subkey = jax.random.split(rng)

        # mc_probs shape: (num_mc_samples, batch, num_classes)
        mc_probs = mc_dropout_forward(
            apply_fn, params, subkey, batch,
            num_samples=num_mc_samples, return_logits=False
        )

        mean_probs, entropy, mutual_info = compute_uncertainty(mc_probs)

        if acquisition_name == 'bald':
            scores = acquisition_bald(entropy, mutual_info)
        elif acquisition_name == 'max_entropy':
            scores = acquisition_max_entropy(entropy, mutual_info)
        elif acquisition_name == 'variation_ratios':
            scores = acquisition_variation_ratios(mean_probs)
        elif acquisition_name == 'mean_std':
            scores = acquisition_mean_std(mc_probs)
        elif acquisition_name == 'random':
            rng, subkey2 = jax.random.split(rng)
            scores = acquisition_random(entropy, subkey2)
        else:
            raise ValueError(f"Unknown acquisition: {acquisition_name}. "
                             f"Choose from {ACQUISITION_FUNCTIONS}")

        all_scores.append(np.array(scores))

    return np.concatenate(all_scores, axis=0)   # (pool_size,)


# ---------------------------------------------------------------------------
# Main active learning loop
# ---------------------------------------------------------------------------

def active_learning_loop(
    pool,
    train_set,
    test_set,
    model,
    rng,
    acquisition_name   = 'bald',
    n_acquisitions     = 10,
    n_steps            = 100,
    num_mc_samples     = 10,
    learning_rate      = 1e-3,
    weight_decay       = 1e-4,
    n_epochs           = 50,
    batch_size         = 128,
    input_shape        = (1, 28, 28, 1),
    reset_model        = True,
    verbose            = True,
):
    """
    Full active learning loop as described in Gal et al. 2017.

    The loop:
      1. (Re)initialise model weights — paper resets at every step to isolate
         the acquisition function's contribution.
      2. Train on the current labelled set until convergence.
      3. Score all pool points with the acquisition function using MC dropout.
      4. Select the top-n most informative points.
      5. Move them from pool → train set (labels come from the dataset,
         simulating a perfect oracle).
      6. Log test accuracy and repeat.

    Args:
        pool             : dict {'image', 'label'} — unlabelled pool
        train_set        : dict {'image', 'label'} — initial labelled set
        test_set         : dict {'image', 'label'} — held-out test set
        model            : BayesianCNN instance from network.py
        rng              : JAX PRNG key
        acquisition_name : which acquisition function to use
        n_acquisitions   : points to acquire per step (paper: 10)
        n_steps          : number of acquisition rounds (paper: 100)
        num_mc_samples   : MC dropout samples for scoring (paper: 10)
        learning_rate    : AdamW learning rate
        weight_decay     : L2 regularisation (≈ prior precision in Bayesian view)
        n_epochs         : training epochs per acquisition step
        batch_size       : training batch size
        input_shape      : shape for parameter initialisation
        reset_model      : if True, reinitialise weights at every step
        verbose          : print progress

    Returns:
        history : list of dicts with keys:
                  'step', 'n_labelled', 'test_accuracy', 'test_loss'
    """
    assert acquisition_name in ACQUISITION_FUNCTIONS, (
        f"acquisition_name must be one of {ACQUISITION_FUNCTIONS}"
    )

    history = []

    # Keep a reference init rng for resetting the model
    rng, init_rng = jax.random.split(rng)
    state = create_train_state(init_rng, model, learning_rate, weight_decay, input_shape)

    if verbose:
        print(f"Active learning | acquisition: {acquisition_name}")
        print(f"Initial labelled set size: {len(train_set['label'])}")
        print(f"Pool size: {len(pool['label'])}")
        print(f"Steps: {n_steps} × {n_acquisitions} acquisitions = "
              f"{n_steps * n_acquisitions} total labels\n")

    for step in range(n_steps):

        # ------------------------------------------------------------------
        # 1. Optionally reset model weights
        # ------------------------------------------------------------------
        if reset_model:
            rng, init_rng = jax.random.split(rng)
            state = create_train_state(
                init_rng, model, learning_rate, weight_decay, input_shape
            )

        # ------------------------------------------------------------------
        # 2. Train on current labelled set
        # ------------------------------------------------------------------
        for epoch in range(n_epochs):
            rng, epoch_rng = jax.random.split(rng)
            state, train_metrics, rng = train_epoch(
                state, train_set, batch_size, epoch_rng
            )

        # ------------------------------------------------------------------
        # 3. Evaluate on test set
        # ------------------------------------------------------------------
        test_metrics = evaluate(state, test_set, batch_size)
        n_labelled   = len(train_set['label'])

        history.append({
            'step'         : step,
            'n_labelled'   : n_labelled,
            'test_accuracy': float(test_metrics['accuracy']),
            'test_loss'    : float(test_metrics['loss']),
        })

        if verbose:
            print(f"Step {step+1:>3}/{n_steps} | "
                  f"labelled: {n_labelled:>5} | "
                  f"test acc: {test_metrics['accuracy']*100:.2f}%")

        # ------------------------------------------------------------------
        # 4. Score pool points with acquisition function
        # ------------------------------------------------------------------
        if len(pool['image']) == 0:
            print("Pool exhausted — stopping early.")
            break

        rng, score_rng = jax.random.split(rng)
        scores = score_pool(
            state.apply_fn, state.params, score_rng,
            pool['image'], acquisition_name, num_mc_samples
        )

        # ------------------------------------------------------------------
        # 5. Select top-n indices (argsort ascending → take last n)
        # ------------------------------------------------------------------
        n_acquire    = min(n_acquisitions, len(pool['image']))
        top_indices  = np.argsort(scores)[-n_acquire:]   # numpy for safe indexing

        # ------------------------------------------------------------------
        # 6. Move selected points from pool → train set
        # ------------------------------------------------------------------
        train_set = append_to_train(train_set, pool, top_indices)
        pool      = remove_from_pool(pool, top_indices)

    return history


# ---------------------------------------------------------------------------
# Convenience: run all acquisition functions and collect results
# ---------------------------------------------------------------------------

def run_all_acquisitions(
    dataset,
    test_set,
    model,
    rng,
    n_per_class    = 2,
    acquisitions   = None,
    **loop_kwargs
):
    """
    Runs the active learning loop once per acquisition function and
    returns all histories for plotting (reproduces Figure 1 of the paper).

    Args:
        dataset      : full training dict {'image', 'label'}
        test_set     : held-out test dict
        model        : BayesianCNN instance
        rng          : JAX PRNG key
        n_per_class  : initial labelled examples per class (paper: 2)
        acquisitions : list of acquisition names to run (default: all)

    Returns:
        results : dict mapping acquisition_name → history list
    """
    if acquisitions is None:
        acquisitions = ACQUISITION_FUNCTIONS

    results = {}
    for acq in acquisitions:
        print(f"\n{'='*50}")
        print(f"Running acquisition: {acq}")
        print(f"{'='*50}")

        # Fresh pool and train set for every acquisition function
        np_rng = np.random.default_rng(42)
        train_set, pool = get_balanced_initial_set(
            dataset, n_per_class=n_per_class, rng=np_rng
        )

        rng, loop_rng = jax.random.split(rng)
        history = active_learning_loop(
            pool, train_set, test_set, model,
            loop_rng, acquisition_name=acq,
            **loop_kwargs
        )
        results[acq] = history

    return results