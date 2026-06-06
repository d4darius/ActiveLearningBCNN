import os
import json
import argparse
import jax
import jax.numpy as jnp
import numpy as np
from libs.network import load_pretrained_vgg16
from libs.isic_loader import load_isic_data, get_isic_splits
from model.learning_loop import active_learning_loop, average_histories

def get_isic_initial_set(pool, rng):
    """
    Samples initial training set of 80 negative and 20 positive examples.
    """
    labels = np.array(pool['label'])
    images = np.array(pool['image'])
    
    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]
    
    # Check if we have enough
    assert len(pos_idx) >= 20, "Not enough positive examples in pool"
    assert len(neg_idx) >= 80, "Not enough negative examples in pool"
    
    chosen_pos = rng.choice(pos_idx, size=20, replace=False)
    chosen_neg = rng.choice(neg_idx, size=80, replace=False)
    
    selected = np.concatenate([chosen_pos, chosen_neg])
    
    mask = np.ones(len(labels), dtype=bool)
    mask[selected] = False
    
    init_train = {
        'image': jnp.array(images[selected]),
        'label': jnp.array(labels[selected]),
    }
    new_pool = {
        'image': jnp.array(images[mask]),
        'label': jnp.array(labels[mask]),
    }
    
    return init_train, new_pool
def run_isic_acquisitions(
    dataset,
    model,
    rng,
    acquisitions,
    n_splits=2,
    n_reps=3,
    **loop_kwargs
):
    all_splits_results = []
    for split in range(n_splits):
        print(f"\n{'#'*50}")
        print(f"Running Split {split+1}/{n_splits}")
        print(f"{'#'*50}")
        
        # 1. Create split (Test set: 100 pos / 100 neg, Pool: rest)
        # We vary the seed for the split to get two *different* random splits.
        test_set, current_pool = get_isic_splits(dataset, seed=42 + split)
        
        # We will average the results over the repetitions for this split
        all_histories = {acq: [] for acq in acquisitions}
        
        for rep in range(n_reps):
            for acq in acquisitions:
                print(f"\n{'='*50}")
                print(f"Split {split+1} | Repetition {rep+1} | Acquisition: {acq}")
                print(f"{'='*50}")
                
                # Reshuffle the pool anew to get a different initial training set
                np_rng = np.random.default_rng(42 + split * 10 + rep)
                train_set, loop_pool = get_isic_initial_set(current_pool, np_rng)
                
                rng, loop_rng = jax.random.split(rng)
                history = active_learning_loop(
                    loop_pool, train_set, test_set, model,
                    loop_rng, acquisition_name=acq,
                    **loop_kwargs
                )
                all_histories[acq].append(history)
                
        # Average results for this split
        split_results = {acq: average_histories(all_histories[acq]) for acq in acquisitions}
        all_splits_results.append(split_results)
        
    return all_splits_results
def main():
    parser = argparse.ArgumentParser(description='ISIC 2016 BALD Active Learning Reproduction')
    parser.add_argument('--output_dir', type=str, default='results/ISIC', help='Output directory')
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    print("Loading ISIC dataset...")
    # This will load images to 224x224 and normalize them
    dataset = load_isic_data()
    if dataset is None:
        print("Failed to load dataset.")
        return
        
    print(f"Dataset loaded: {len(dataset['image'])} total images.")
    # Model and keys
    rng = jax.random.PRNGKey(0)
    rng, init_rng = jax.random.split(rng)
    
    print("Loading Pre-trained VGG16 model...")
    # This also initializes Flax model and loads PyTorch weights
    vgg16_params, model = load_pretrained_vgg16(init_rng)
    # Note: the `active_learning_loop` expects `model` and will initialize its own weights
    # if `reset_model` is True, it calls `create_train_state`.
    # Wait: create_train_state initializes weights from scratch!
    # I need to modify create_train_state or pass the pre-trained weights to `active_learning_loop`.
    # In `learning_loop.py`, `create_train_state` does `params = model.init(...)`.
    # To properly reset to *pre-trained* weights, we should pass the loaded params instead.
    
    # We will override the `create_train_state` locally for this script so we don't break MNIST.
    pass
    
    loop_kwargs = dict(
        n_steps         = 8, # 100 acquisitions per step until pool exhausted (~700 points total) -> 7 steps of 100.
        n_acquisitions  = 100, # acquire 100 at a time
        num_mc_samples  = 20, # Paper states MC dropout with 20 samples for ISIC
        learning_rate   = 1e-5, # small learning rate for VGG16 fine-tuning
        weight_decay    = 0.0, # Will be set dynamically in learning_loop.py
        n_epochs        = 100, # 100 epochs until convergence
        batch_size      = 8,
        reset_model     = True,
        input_shape     = (1, 224, 224, 3),
    )
    
    acquisitions = ['random', 'bald']
    
    # We need to inject the pre-trained params into active_learning_loop.
    # To avoid changing the original files too much, we will patch `create_train_state` in the `learning_loop` module.
    import model.learning_loop as ll
    import optax
    from flax.training import train_state
    
    original_create_train_state = ll.create_train_state
    
    def custom_create_train_state(rng, model, learning_rate, weight_decay, input_shape):
        # Use AdamW and the *pre-trained* params instead of random init
        tx = optax.adamw(learning_rate, weight_decay=weight_decay)
        # vgg16_params contains the loaded weights
        return train_state.TrainState.create(apply_fn=model.apply, params=vgg16_params, tx=tx)
        
    ll.create_train_state = custom_create_train_state
    all_splits_results = run_isic_acquisitions(
        dataset,
        model,
        rng,
        acquisitions,
        n_splits=2,
        n_reps=3,
        **loop_kwargs
    )
    
    # Save results
    json_path = os.path.join(args.output_dir, "isic_results.json")
    with open(json_path, 'w') as f:
        json.dump(all_splits_results, f, indent=2)
        
    print(f"\nResults saved to {json_path}")
if __name__ == '__main__':
    main()