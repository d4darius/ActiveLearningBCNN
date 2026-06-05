#!/bin/bash

# Move to the project root directory
cd "$(dirname "$0")/.." || exit

echo "==============================================================="
echo "Reproducing Figure 1: All Acquisition Functions"
echo "==============================================================="

# This runs the Bayesian CNN (default) against all implemented 
# acquisition functions (BALD, Max Entropy, Variation Ratios, Mean STD, Random)
# to reproduce the 'all_accuracy.png' plot as seen in the original paper.

# The parameters match the paper's experimental setup:
# - Initial training set: 20 points
# - Acquisitions: 100 steps of 10 points each
COMMON_ARGS="--n_steps 100 --n_acquisitions 10 --n_per_class 2"

echo ">>> Running evaluation for ALL acquisition functions <<<"
python main.py --model_type bayesian --acquisition all $COMMON_ARGS

echo ""
echo "==============================================================="
echo "Complete! Plot is saved in 'results/all_bayesian_accuracy.png'"
echo "==============================================================="
