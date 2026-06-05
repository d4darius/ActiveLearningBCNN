# Deep Bayesian Active Learning with Image Data

This repository aims at reproducing the experimental results of the paper ["Deep Bayesian Active Learning with Image Data"](https://arxiv.org/abs/1703.02910) using JAX and Flax. 

By leveraging the speed and vectorization capabilities of JAX, this project implements a framework for training Bayesian Convolutional Neural Networks (BCNNs) and utilizing their uncertainty estimates to perform efficient active learning.

## 🎯 Project Objectives

As outlined in the original scope, this repository is built to:
* Provide a short introduction to the claims and theory behind Bayesian Active Learning.
* Reproduce experimental results by training and evaluating Bayesian CNNs.
* Implement various active learning strategies.
* Evaluate and compare the performance of these active learning strategies over time.

---

## 🚀 Setup & Installation

We recommend using a Conda environment to manage dependencies:

```bash
conda create -n bcnn-al python=3.10
conda activate bcnn-al
pip install -r requirements.txt
```

*Note: For GPU support, check the instructions in `requirements.txt` to install the correct `jaxlib` for your CUDA version.*

---

## 🧪 Experiments

All scripts designed to automatically reproduce the experiments from the paper are located in the `experiments` directory. You can run these bash scripts to launch evaluations across different acquisition functions and model configurations.

---

## 🛠️ Implementation Details

The codebase relies on Monte Carlo (MC) Dropout to approximate Bayesian inference. 

### Core Components

* **Bayesian CNN (`network.py`)**: Implements a Flax-based Convolutional Neural Network with interleaved Dropout layers. By keeping the `deterministic=False` flag active during inference, we can sample from the approximate posterior.
* **Loss Formulation (`loss.py`)**: Uses standard Cross-Entropy Loss alongside L2 regularization (weight decay). In the context of MC Dropout, this combination is mathematically equivalent to minimizing the Kullback-Leibler (KL) divergence between the approximate and true posterior. An alternative NumPyro-based negative log-likelihood loss is also included.
* **Training Loop (`train.py`)**: Utilizes Optax's `adamw` optimizer to properly handle weight decay, which corresponds to the prior variance in the Bayesian formulation.
* **Uncertainty Estimation (`utils.py`)**: Employs `jax.vmap` to efficiently parallelize stochastic forward passes across multiple PRNG keys. 

### Uncertainty Metrics

The repository calculates two primary uncertainty metrics using the MC Dropout samples:

1.  **Predictive Entropy (Total Uncertainty)**: 
    Calculated as $H(y|x) = -\sum_{c}p(y=c|x)\log p(y=c|x)$.
2.  **Mutual Information (Epistemic Uncertainty)**:
    Calculated as $I(y, w|x) = H(y|x) - \mathbb{E}[H(y|x, w)]$, representing the uncertainty captured by the model weights.

---

## 📂 Repository Structure

| File | Description |
| :--- | :--- |
| **`network.py`** | Contains the `BayesianCNN` Flax module with configurable dropout probabilities and classes. |
| **`loss.py`** | Defines standard and NumPyro-based loss functions, along with accuracy metrics. |
| **`utils.py`** | Houses the `mc_dropout_forward` function and the math for calculating entropy and mutual information. |
| **`train.py`** | Implements the JAX-compiled `train_step`, `eval_step`, and the epoch loops. |


---

## 🔬 ISIC 2016 Experiment (Part 3B)
We also reproduce the second experiment from the paper, which involves applying the active learning pipeline to a smaller, unbalanced medical dataset: **ISBI 2016 Skin Lesion Analysis Towards Melanoma Detection**.

### Setup and Running

1. **Dependencies**: 
   Ensure you have installed the additional dependencies required for data processing and metric calculation:
   ```bash
   pip install pandas scikit-learn opencv-python requests
   ```
2. **Run the Experiment**:
   We have created a dedicated main script `main_isic.py` which handles:
   - Downloading and preparing the dataset (~900 images).
   - Loading pre-trained `torchvision` weights into a custom JAX/Flax VGG16 model with MC Dropout.
   - Using dynamic data augmentation (random flips) for the positive (malignant) examples during training.
   - Logging the Area-Under-the-Curve (AUC) and the number of positive samples acquired.
   Run the script with:
   ```bash
   python main_isic.py --output_dir results_isic
   ```
3. **Plotting**:
   To generate the plots for AUC and Positive Samples Acquired vs. Number of Acquisitions:
   ```bash
   python plot_isic.py --input_json results_isic/isic_results.json --output_dir results_isic
   ```