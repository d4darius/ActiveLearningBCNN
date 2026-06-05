import json
import os
import matplotlib.pyplot as plt
import argparse
import numpy as np

def plot_split_results(split_results, split_idx, output_dir):
    """
    Plots AUC vs number of acquired images and positive samples vs acquired images for a single split side-by-side.
    """
    COLORS = {
        'bald'  : 'blue',
        'random': 'green',
    }
    LABELS = {
        'bald'  : 'BALD',
        'random': 'Uniform',
    }
    
    # Create a single figure with 2 subplots side-by-side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. AUC Plot (Left)
    for acq, history in split_results.items():
        # steps start with initial set (n_labelled = 100), then +100 each time
        # the x axis is 'Number of acquired images' which means n_labelled - 100
        x = np.array([h['n_labelled'] - 100 for h in history])
        y = np.array([h['test_auc'] for h in history])
        y_err = np.array([h.get('test_auc_std', 0.0) for h in history])
        
        ax1.plot(x, y,
                 label=LABELS.get(acq, acq),
                 color=COLORS.get(acq, 'black'),
                 linewidth=2, marker='s')
                 
        ax1.fill_between(x, y - y_err, y + y_err, color=COLORS.get(acq, 'black'), alpha=0.2)
                 
    ax1.set_xlabel('Number of acquired images', fontsize=12)
    ax1.set_ylabel('Test AUC', fontsize=12)
    ax1.set_title(f'Test AUC vs. Acquisitions (Split {split_idx})', fontsize=13)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # 2. Positive Samples Plot (Right)
    for acq, history in split_results.items():
        x = np.array([h['n_labelled'] - 100 for h in history])
        y = np.array([h['n_pos_labelled'] - 20 for h in history]) # Subtract the initial 20 pos samples
        y_err = np.array([h.get('n_pos_labelled_std', 0.0) for h in history])
        
        ax2.plot(x, y,
                 label=LABELS.get(acq, acq),
                 color=COLORS.get(acq, 'black'),
                 linewidth=2, marker='s')
                 
        ax2.fill_between(x, y - y_err, y + y_err, color=COLORS.get(acq, 'black'), alpha=0.2)
                 
    ax2.set_xlabel('Number of acquired images', fontsize=12)
    ax2.set_ylabel('Number of positive examples acquired', fontsize=12)
    ax2.set_title(f'Positive Examples vs. Acquisitions (Split {split_idx})', fontsize=13)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    # Adjust layout and save the combined figure
    plt.tight_layout()
    combined_path = os.path.join(output_dir, f"split_{split_idx}_combined.png")
    plt.savefig(combined_path, dpi=150)
    plt.close()
    
    print(f"Saved combined plot for split {split_idx} to {combined_path}")

def main():
    parser = argparse.ArgumentParser(description='Plot ISIC 2016 results')
    parser.add_argument('--input_json', type=str, default='results/ISIC/isic_results.json', help='Path to results JSON')
    parser.add_argument('--output_dir', type=str, default='results/ISIC', help='Output directory for plots')
    args = parser.parse_args()
    
    if not os.path.exists(args.input_json):
        print(f"Error: {args.input_json} does not exist.")
        return
        
    os.makedirs(args.output_dir, exist_ok=True)
    
    with open(args.input_json, 'r') as f:
        all_splits_results = json.load(f)
        
    for i, split_results in enumerate(all_splits_results):
        plot_split_results(split_results, i + 1, args.output_dir)

if __name__ == '__main__':
    main()
