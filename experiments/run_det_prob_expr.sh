#!/bin/bash

# Move to the project root directory
cd "$(dirname "$0")/.." || exit

echo "==============================================================="
echo "Starting Evaluation: Bayesian vs Deterministic"
echo "Acquisition Functions: BALD, Variation Ratios, Max Entropy"
echo "==============================================================="

# Common arguments for both runs matching the paper's experimental setup:
# - Initial training set: 20 points (2 per class for 10 classes)
# - Acquisitions: 100 steps
# - Points per acquisition: 10
# - Repetitions: 3 (handled internally by the Python script)
COMMON_ARGS="--n_steps 100 --n_acquisitions 10 --n_per_class 2 --acquisition bald variation_ratios max_entropy"

echo ""
echo ">>> Running Bayesian Model Evaluation <<<"
python main.py --model_type bayesian $COMMON_ARGS

echo ""
echo ">>> Running Deterministic Model Evaluation <<<"
python main.py --model_type deterministic $COMMON_ARGS

echo ""
echo ">>> Generating Comparison Plot <<<"
python experiments/comparison_det_prob.py --dir results --output results/deterministic_vs_bayesian_plot.png

echo ""
echo "==============================================================="
echo "Evaluation Complete!"
echo "Data JSONs and the comparison plot are saved in the 'results' folder."
echo "==============================================================="
