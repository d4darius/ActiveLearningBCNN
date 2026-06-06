import argparse
import json
import os

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import torchvision


from libs.network import BayesianCNN
from model.learning_loop import (
    ACQUISITION_FUNCTIONS,
    active_learning_loop,
    get_balanced_initial_set,
    run_acquisitions,
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_mnist():
    """Downloads and prepares MNIST using torchvision."""
    def to_dict(dataset):
        images = np.stack([
            np.array(img, dtype=np.float32) / 255.0
            for img, _ in dataset
        ])                                          # (N, 28, 28)
        images = images[..., np.newaxis]            # (N, 28, 28, 1)
        labels = np.array([lbl for _, lbl in dataset], dtype=np.int32)
        return {'image': jnp.array(images), 'label': jnp.array(labels)}

    train = torchvision.datasets.MNIST(root='./data', train=True,  download=True)
    test  = torchvision.datasets.MNIST(root='./data', train=False, download=True)

    return to_dict(train), to_dict(test)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_history(results, save_path=None):
    """
    Plots test accuracy vs number of labelled images.
    results: dict mapping acquisition_name -> history list
             (as returned by run_all_acquisitions or a single loop)
    """
    COLORS = {
        'bald'            : '#3266ad',
        'max_entropy'     : '#993c1d',
        'variation_ratios': '#0f6e56',
        'mean_std'        : '#854f0b',
        'random'          : '#73726c',
    }
    LABELS = {
        'bald'            : 'BALD',
        'max_entropy'     : 'Max Entropy',
        'variation_ratios': 'Variation Ratios',
        'mean_std'        : 'Mean STD',
        'random'          : 'Random',
    }

    plt.figure(figsize=(8, 5))
    for acq, history in results.items():
        x = [h['n_labelled']    for h in history]
        y = [h['test_accuracy'] * 100 for h in history]
        plt.plot(x, y,
                 label=LABELS.get(acq, acq),
                 color=COLORS.get(acq, None),
                 linewidth=2)

    plt.xlabel('Number of labelled images', fontsize=12)
    plt.ylabel('Test accuracy (%)', fontsize=12)
    plt.title('Deep Bayesian Active Learning — MNIST', fontsize=13)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description='Deep Bayesian Active Learning — MNIST reproduction'
    )

    # --- Acquisition ---
    parser.add_argument(
        '--acquisition', '-a',
        type=str,
        nargs='+',
        default=['bald'],
        choices=ACQUISITION_FUNCTIONS + ['all'],
        help=(
            'One or more acquisition functions to run. '
            'Use "all" to run every function (reproduces Figure 1 of the paper). '
            'Results are plotted together for easy comparison. '
            f'Choices: {ACQUISITION_FUNCTIONS + ["all"]}. '
            'Default: bald. '
            'Examples: '
            '--acquisition bald random | '
            '--acquisition bald max_entropy variation_ratios | '
            '--acquisition all'
        )
    )

    # --- Active learning loop ---
    parser.add_argument(
        '--n_steps',
        type=int,
        default=100,
        help='Number of acquisition rounds (default: 100)'
    )
    parser.add_argument(
        '--n_acquisitions',
        type=int,
        default=10,
        help='Images acquired from pool per step (default: 10)'
    )
    parser.add_argument(
        '--n_per_class',
        type=int,
        default=2,
        help='Initial labelled examples per class (default: 2, total=20)'
    )
    parser.add_argument(
        '--num_mc_samples',
        type=int,
        default=10,
        help='MC dropout forward passes for uncertainty estimation (default: 10)'
    )

    # --- Training ---
    parser.add_argument(
        '--n_epochs',
        type=int,
        default=200,
        help='Max training epochs per acquisition step (default: 200, with early stopping)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=128,
        help='Training batch size (default: 128)'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=1e-3,
        help='AdamW learning rate (default: 1e-3)'
    )
    parser.add_argument(
        '--weight_decay',
        type=float,
        default=1e-4,
        help=(
            'L2 regularisation — corresponds to the prior precision '
            'in the Bayesian formulation (default: 1e-4)'
        )
    )
    parser.add_argument(
        '--no_reset',
        action='store_true',
        help=(
            'Do NOT reset model weights between acquisition steps. '
            'The paper always resets (default behaviour) to isolate '
            'the acquisition function effect.'
        )
    )
    parser.add_argument(
        '--patience',
        type=int,
        default=10,
        help=(
            'Early stopping patience: stop training if loss does not improve '
            'for this many consecutive epochs (default: 10)'
        )
    )
    parser.add_argument(
        '--min_delta',
        type=float,
        default=1e-4,
        help=(
            'Minimum loss improvement to count as progress for early stopping '
            '(default: 1e-4)'
        )
    )

    # --- Model ---
    parser.add_argument(
        '--dropout_prob',
        type=float,
        default=0.5,
        help='Dropout probability for the Bayesian CNN (default: 0.5)'
    )

    # --- Model Type ---
    parser.add_argument(
        '--model_type',
        type=str,
        default='bayesian',
        choices=['bayesian', 'deterministic'],
        help='Whether to run a Bayesian CNN (dropout enabled) or Deterministic CNN (dropout disabled everywhere).'
    )

    # --- Reproducibility ---
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help='JAX PRNG seed (default: 0)'
    )

    # --- Output ---
    parser.add_argument(
        '--output_dir',
        type=str,
        default='results/MNIST',
        help='Directory to save plots and history JSON (default: results/)'
    )
    parser.add_argument(
        '--no_plot',
        action='store_true',
        help='Skip plotting (useful when running on a remote server)'
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # --- Setup output directory ---
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Print config ---
    print("\n=== Configuration ===")
    for k, v in vars(args).items():
        print(f"  {k:<20}: {v}")
    print()

    # --- Load data ---
    print("Loading MNIST...")
    mnist_train, mnist_test = load_mnist()
    print(f"Train: {mnist_train['image'].shape} | Test: {mnist_test['image'].shape}\n")

    # --- Model ---
    model = BayesianCNN(num_classes=10, dropout_prob=args.dropout_prob)
    rng   = jax.random.PRNGKey(args.seed)

    # --- Common loop kwargs ---
    loop_kwargs = dict(
        n_steps         = args.n_steps,
        n_acquisitions  = args.n_acquisitions,
        num_mc_samples  = args.num_mc_samples,
        learning_rate   = args.learning_rate,
        weight_decay    = args.weight_decay,
        n_epochs        = args.n_epochs,
        batch_size      = args.batch_size,
        reset_model     = not args.no_reset,
        input_shape     = (1, 28, 28, 1),
        patience        = args.patience,
        min_delta       = args.min_delta,
    )

    # --- Run ---
    # if args.acquisition == ['all']:
    #     # Reproduces Figure 1 of the paper
    #     results = run_acquisitions(
    #         dataset     = mnist_train,
    #         test_set    = mnist_test,
    #         model       = model,
    #         rng         = rng,
    #         n_per_class = args.n_per_class,
    #         **loop_kwargs,
    #     )
    # else:
    # Single acquisition function
    results = run_acquisitions(
        dataset      = mnist_train,
        test_set     = mnist_test,
        model        = model,
        rng          = rng,
        n_per_class  = args.n_per_class,
        acquisitions = args.acquisition,
        is_deterministic = (args.model_type == 'deterministic'),
        **loop_kwargs,
    )

    file_name = "_".join(map(str, args.acquisition))
    # --- Save history as JSON ---
    json_path = os.path.join(args.output_dir, f"{file_name}_{args.model_type}_history.json")
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nHistory saved to {json_path}")

    # --- Plot ---
    if not args.no_plot:
        plot_path = os.path.join(args.output_dir, f"{file_name}_{args.model_type}_accuracy.png")
        plot_history(results, save_path=plot_path)


if __name__ == '__main__':
    main()