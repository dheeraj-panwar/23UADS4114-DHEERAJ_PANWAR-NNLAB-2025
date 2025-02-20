# Perceptron Logic Gate Classifier

## Object
WAP to implement the Perceptron Learning Algorithm using numpy in Python. Evaluate performance of a single perceptron for NAND and XOR truth tables as input dataset.**.

## Model Explanation
- **Perceptron Class**: Initializes weights, applies activation function, trains using weight updates.
- **Training on NAND Gate**: Learns successfully since NAND is **linearly separable**.
- **Training on XOR Gate**: Fails since XOR is **not linearly separable** (needs a multi-layer perceptron).

## Code Explanation

### Perceptron Class
- Initializes random weights (including bias).
- Uses a **step activation function** (outputs 1 if weighted sum ≥ 0, else 0).
- Trains by updating weights using the rule:

- Evaluates model accuracy by comparing predictions with actual values.

### Training on NAND Gate
- Provides input-output pairs for the NAND logic gate.
- The perceptron successfully learns and achieves high accuracy.

### Training on XOR Gate
- Provides input-output pairs for the XOR logic gate.
- The perceptron **fails** since XOR is not linearly separable.

## Results and Performance
| Gate  | Accuracy (%) | Predictions  | Expected Output | Linearly Separable? |
|--------|------------|-------------|----------------|----------------------|
| **NAND** | ~100.00%  | `[1, 1, 1, 0]`  | `[1, 1, 1, 0]`  | ✅ Yes  |
| **XOR**  | ~50.00%   | `[0, 0, 0, 0]` (or incorrect) | `[0, 1, 1, 0]`  | ❌ No  |

## Limitations
- A **single-layer perceptron** can only learn linearly separable functions.
- **XOR cannot be learned** without adding hidden layers (MLP required).
