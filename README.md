# Single-Neuron Artificial Neural Network from Scratch in NumPy

A transparent, first-principles implementation of a single-neuron feedforward artificial neural network for non-linear binary classification, built purely in Python using NumPy without external deep learning frameworks.

---

## Overview

Modern deep learning frameworks often abstract the foundational mathematics governing neural computations. This repository reconstructs the complete forward and backward propagation mechanics from scratch:
- Direct computation of weighted sums and bias offsets
- Logistic sigmoid non-linear activation
- Analytical gradient backpropagation via the multivariate Chain Rule
- Parameter optimization through Stochastic Gradient Descent (SGD)
- 2D decision boundary visualization separating class manifolds

---

## Mathematical Formulation

### 1. Network Topology

The model consists of a single artificial neuron with two continuous input features and a scalar output:

- **Inputs**: $x = [x_1, x_2]^T$ (Petal Length, Petal Width)
- **Parameters**: Weights $w = [w_1, w_2]^T$ and scalar bias $b$
- **Pre-Activation Function**:
  $$z = w^T x + b = w_1 x_1 + w_2 x_2 + b$$
- **Activation Function (Logistic Sigmoid)**:
  $$\hat{y} = \sigma(z) = \frac{1}{1 + e^{-z}}$$

### 2. Loss Function

Training optimizes parameters by minimizing the squared error loss over individual observations:
$$C = (\hat{y} - y)^2$$

where $y \in \{0, 1\}$ denotes the ground-truth binary class label.

### 3. Analytical Backpropagation (Chain Rule)

The gradients of the cost function with respect to model parameters are derived analytically:

1. **Derivative of Loss with respect to Prediction**:
   $$\frac{\partial C}{\partial \hat{y}} = 2(\hat{y} - y)$$

2. **Derivative of Sigmoid with respect to Pre-activation ($z$)**:
   $$\frac{\partial \hat{y}}{\partial z} = \sigma'(z) = \sigma(z)(1 - \sigma(z)) = \hat{y}(1 - \hat{y})$$

3. **Combined Error Term ($\delta$)**:
   $$\delta = \frac{\partial C}{\partial z} = \frac{\partial C}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z} = 2(\hat{y} - y) \cdot \sigma'(z)$$

4. **Parameter Gradients**:
   $$\frac{\partial C}{\partial w_1} = \delta \cdot \frac{\partial z}{\partial w_1} = \delta \cdot x_1$$
   $$\frac{\partial C}{\partial w_2} = \delta \cdot \frac{\partial z}{\partial w_2} = \delta \cdot x_2$$
   $$\frac{\partial C}{\partial b} = \delta \cdot \frac{\partial z}{\partial b} = \delta \cdot 1$$

### 4. Parameter Update Rule

Weights and bias are iteratively adjusted via Stochastic Gradient Descent:
$$w_1 \leftarrow w_1 - \alpha \frac{\partial C}{\partial w_1}$$
$$w_2 \leftarrow w_2 - \alpha \frac{\partial C}{\partial w_2}$$
$$b \leftarrow b - \alpha \frac{\partial C}{\partial b}$$

where $\alpha$ denotes the learning rate (step size).

---

## Repository Structure

```
neural-network-from-scratch/
├── notebooks/
│   └── flower_neural_network.ipynb   # Interactive Jupyter notebook with complete walkthrough
├── src/
│   ├── __init__.py                   # Package initializer
│   ├── activation.py                 # Sigmoid and analytical derivative functions
│   ├── model.py                      # SingleNeuronClassifier class with forward & SGD backprop
│   └── visualization.py              # Loss curves and 2D decision boundary plotting
├── .gitignore                        # Python and Jupyter ignore patterns
├── LICENSE                           # MIT Open Source License
├── main.py                           # Command-line interface execution pipeline
└── requirements.txt                  # Python runtime dependencies
```

---

## Dataset Description

The binary classification task distinguishes between two flower categories based on petal dimensions:

| Sample Index | Petal Length ($x_1$) | Petal Width ($x_2$) | Ground Truth Class ($y$) | Label Interpretation |
| :---: | :---: | :---: | :---: | :---: |
| 1 | 3.0 | 1.5 | 1 | Class 1 (Red) |
| 2 | 2.0 | 1.0 | 0 | Class 0 (Blue) |
| 3 | 4.0 | 1.5 | 1 | Class 1 (Red) |
| 4 | 3.0 | 1.0 | 0 | Class 0 (Blue) |
| 5 | 3.5 | 0.5 | 1 | Class 1 (Red) |
| 6 | 2.0 | 0.5 | 0 | Class 0 (Blue) |
| 7 | 5.5 | 1.0 | 1 | Class 1 (Red) |
| 8 | 1.0 | 1.0 | 0 | Class 0 (Blue) |

**Unlabelled Query Point**: Length = 4.5, Width = 1.0.

---

## Installation & Setup

### Prerequisites

- Python 3.8+
- `pip` package manager

### Environment Configuration

1. Clone the repository:
   ```bash
   git clone https://github.com/0xShashankbtc/neural-network-from-scratch.git
   cd neural-network-from-scratch
   ```

2. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### 1. Interactive Notebook

Launch JupyterLab or Jupyter Notebook:
```bash
jupyter notebook notebooks/flower_neural_network.ipynb
```

### 2. Standalone CLI Execution

Run the complete pipeline from terminal:
```bash
python3 main.py
```

### Sample Output:
```
============================================================
SINGLE-NEURON NEURAL NETWORK FROM SCRATCH IN NUMPY
============================================================
...
Trained Parameters: w1=3.3110, w2=1.2945, b=-10.8249
Training Set Accuracy: 87.5% (7/8)

Mystery Sample Classification:
  Query Point: Petal Length = 4.5, Petal Width = 1.0
  Raw Activation Output (Sigmoid Probability): 0.9954
  Predicted Classification:                   Red Flower (Class 1)
```

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for complete details.
