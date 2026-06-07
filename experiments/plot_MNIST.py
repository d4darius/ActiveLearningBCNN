import matplotlib.pyplot as plt
import json


def plot_history(results, save_path=None):
    """
    Plots test accuracy vs number of labelled images.
    results: dict mapping acquisition_name -> history list
             (as returned by run_all_acquisitions or a single loop)
    """
    COLORS = {
        'bald'            : '#ef5d58', # Coral Red
        'variation_ratios': '#799dc6', # Soft Blue
        'max_entropy'     : '#7bbf84', # Muted Green
        'mean_std'        : '#a395a5', # Greyish Purple
        'random'          : '#e09b80', # Peach/Orange
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
    plt.ylim(bottom=80)
    plt.title('Deep Bayesian Active Learning — MNIST', fontsize=13)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

def compute_acquisition_table(results, error_thresholds=[10.0, 5.0]):
    rows = []
    for threshold in error_thresholds:
        accuracy_target = 1.0 - (threshold / 100.0)  # 10% error → 0.90 accuracy
        row = {'% error': f"{threshold}%"}
        for acq, history in results.items():
            n_labelled = next(
                (entry['n_labelled'] for entry in history 
                 if entry['test_accuracy'] >= accuracy_target),
                'never'
            )
            row[acq] = n_labelled
        rows.append(row)
    return rows


with open('../results/MNIST/all_bayesian_history.json', 'r') as f:
    results_data = json.load(f)

plot_history(results_data, '../results/MNIST/MNIST_plot.png')
# Compute and print
rows = compute_acquisition_table(results_data)

# Print as a formatted table
acq_names = list(results_data.keys())

header = f"{'% error':<10}" + "".join(f"{a:<20}" for a in acq_names)
print(header)
print("-" * len(header))
for row in rows:
    line = f"{row['% error']:<10}" + "".join(f"{str(row[a]):<20}" for a in acq_names)
    print(line)