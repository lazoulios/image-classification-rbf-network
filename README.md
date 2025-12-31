# Radial Basis Function (RBF) Network for CIFAR-10 Classification

The goal of this repository is to implement a **custom Radial Basis Function (RBF) Neural Network** from scratch and evaluate its performance on the CIFAR-10 dataset.

The project constructs a hybrid architecture using **K-Means Clustering** for the hidden layer (finding prototypes) and **Logistic Regression** for the output layer. It compares this approach against distance-based baselines like **k-Nearest Neighbors (k-NN)** and **Nearest Class Centroid**.

**Principal Component Analysis (PCA)** is utilized for dimensionality reduction to make distance calculations computationally feasible.

## 📊 Key Results

The experiments demonstrate that the RBF Network significantly outperforms the baseline methods. Increasing the number of RBF centers (neurons) leads to better generalization, proving that learning "prototypes" is more effective than simple "lazy learning" (k-NN).

| Model Architecture | Configuration | Test Accuracy | Comments |
| :--- | :--- | :--- | :--- |
| **RBF Network** | **500 Centers** | **41.72%** | **Best Performance.** |
| **RBF Network** | 200 Centers | 39.76% | Good balance of speed/accuracy. |
| **RBF Network** | 100 Centers | 38.02% | Fast training. |
| **k-NN** | k=1 (Baseline) | 37.38% | Prone to noise. |
| **Nearest Centroid** | Baseline | 27.75% | Too simple for complex image data. |

-----

## 🚀 Project Setup and Execution

### 1. Requirements

The project uses Python 3.x and relies on the following major libraries:

* `Scikit-learn` (for K-Means, Logistic Regression, PCA, and Scaling)
* `TensorFlow / Keras` (only for loading the CIFAR-10 dataset)
* `Joblib` (for saving/loading trained models)
* `Matplotlib / Seaborn` (for plotting confusion matrices)

### 2. Installation

Clone the repository and set up a virtual environment:

```bash
# Clone the repository
git clone [https://github.com/lazoulios/image-classification-rbf-network](https://github.com/lazoulios/image-classification-rbf-network)
cd image-classification-rbf-network

# Create and activate the virtual environment
python -m venv venv
source venv/bin/activate  # Use venv\Scripts\Activate.ps1 on Windows PowerShell

# Install dependencies from requirements.txt
pip install -r requirements.txt

```

### 3. Running the Models

The project includes a custom class `RBFNetwork` located in `rbf.py`. The execution logic uses a **"Load or Fit"** approach: it checks if a trained model exists in the `models/` folder to avoid redundant training.

#### A. Training and Evaluation

Runs the experimental loop for RBF Networks with **100, 200, and 500 centers**. It applies PCA, trains the models, and prints a comparative results table.

```bash
python main.py

```

#### B. Visualization

Generates visual evidence for the best-performing model (500 Centers), including a **Confusion Matrix** and examples of **Correct vs. Incorrect predictions**.

```bash
python visualize.py

```

---

## 📁 Repository Structure

* `rbf.py`: **Core file.** Contains the custom `RBFNetwork` class implementation (fit/predict logic).
* `main.py`: Main script for running experiments and training the models.
* `visualize.py`: Script for generating plots and visual examples.
* `utils.py`: Helper functions for data loading and model persistence.
* `models/`: Directory where trained `.pkl` models are saved (ignored by Git).
* `images/`: Directory where generated plots are saved.
* `requirements.txt`: List of all required Python packages.
