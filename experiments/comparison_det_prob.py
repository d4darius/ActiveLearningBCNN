import json
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt


def load_json(filepath):
    if not os.path.exists(filepath):
        print(f"Warning: File not found: {filepath}")
        return None
    with open(filepath, 'r') as f:
        return json.load(f)


def plot_model(ax, history, color, label):
    """
    Plots a single model's accuracy curve with std deviation band.

    Args:
        ax      : matplotlib Axes to plot on
        history : list of dicts with 'n_labelled', 'test_accuracy',
                  and optionally 'test_accuracy_std'
        color   : line / band color
        label   : legend label
    """
    x = np.array([h['n_labelled'] for h in history])
    y = np.array([h['test_accuracy'] * 100 for h in history])

    has_std = 'test_accuracy_std' in history[0]
    if has_std:
        y_std = np.array([h['test_accuracy_std'] * 100 for h in history])
    else:
        y_std = np.zeros_like(y)

    ax.plot(x, y, label=label, color=color, linewidth=2)
    if has_std:
        ax.fill_between(x, y - y_std, y + y_std, color=color, alpha=0.18)


def main():
    parser = argparse.ArgumentParser(
        description="Plot comparison between Bayesian and Deterministic runs."
    )
    parser.add_argument(
        '--dir', type=str, default='results',
        help='Directory where JSON histories are saved.'
    )
    parser.add_argument(
        '--output', type=str, default='results/comparison_plot.png',
        help='Output plot file path.'
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Load data for both model types
    # ------------------------------------------------------------------
    bayesian_data = {}
    deterministic_data = {}

    for f in os.listdir(args.dir):
        if f.endswith('_bayesian_history.json'):
            data = load_json(os.path.join(args.dir, f))
            if data:
                for acq, history in data.items():
                    bayesian_data[acq] = history
        elif f.endswith('_deterministic_history.json'):
            data = load_json(os.path.join(args.dir, f))
            if data:
                for acq, history in data.items():
                    deterministic_data[acq] = history

    if not bayesian_data and not deterministic_data:
        print("No history JSON files found in", args.dir)
        return

    LABELS = {
        'bald'            : 'BALD',
        'max_entropy'     : 'Max Entropy',
        'variation_ratios': 'Variation Ratios',
        'mean_std'        : 'Mean STD',
        'random'          : 'Random',
    }

    # Use a stable ordering: keep only acquisition functions present in data
    ORDERED_ACQS = ['bald', 'max_entropy', 'variation_ratios', 'mean_std', 'random']
    all_acqs = set(bayesian_data.keys()).union(set(deterministic_data.keys()))
    acqs = [a for a in ORDERED_ACQS if a in all_acqs]

    if not acqs:
        print("No acquisition functions found in the data.")
        return

    # ------------------------------------------------------------------
    # Plot: one subplot per acquisition function, all in one row
    # ------------------------------------------------------------------
    n_plots = len(acqs)
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5), squeeze=False)
    axes = axes[0]  # flatten to 1-D

    COLOR_BAYESIAN     = '#d62728'   # red
    COLOR_DETERMINISTIC = '#1f77b4'  # blue

    for i, acq in enumerate(acqs):
        ax = axes[i]

        # Plot Bayesian
        if acq in bayesian_data and bayesian_data[acq]:
            plot_model(
                ax, bayesian_data[acq],
                color=COLOR_BAYESIAN,
                label='Bayesian',
            )

        # Plot Deterministic
        if acq in deterministic_data and deterministic_data[acq]:
            plot_model(
                ax, deterministic_data[acq],
                color=COLOR_DETERMINISTIC,
                label='Deterministic',
            )

        ax.set_title(LABELS.get(acq, acq), fontsize=13, fontweight='bold')
        ax.set_xlabel('Number of labelled images', fontsize=11)
        if i == 0:
            ax.set_ylabel('Test accuracy (%)', fontsize=11)
        ax.legend(fontsize=10, loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(70, 100)

    fig.suptitle(
        'Bayesian vs Deterministic — Active Learning on MNIST',
        fontsize=14, fontweight='bold', y=1.02,
    )
    fig.tight_layout()

    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Comparison plot saved to {args.output}")


if __name__ == '__main__':
    main()
