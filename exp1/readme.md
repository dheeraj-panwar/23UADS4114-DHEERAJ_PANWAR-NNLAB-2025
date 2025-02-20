# Experiment 1: Perceptron Learning Algorithm

## Objective
Write a Python program to implement the **Perceptron Learning Algorithm** using NumPy. Evaluate the performance of a **single-layer perceptron** for **NAND and XOR truth tables** as input datasets.

---

## Model Explanation
- **Perceptron Class**: Initializes weights, applies an activation function, and updates weights during training.
- **Training on NAND Gate**: The perceptron learns successfully since **NAND is linearly separable**.
- **Training on XOR Gate**: The perceptron **fails** since **XOR is not linearly separable** (requires a multi-layer perceptron).

---

## Python Code

```python
import numpy as np

class Perceptron:
    def __init__(self, input_size, learning_rate=0.1, epochs=100):
        self.weights = np.random.randn(input_size) * 0.01
        self.bias = 0
        self.learning_rate = learning_rate
        self.epochs = epochs

    def activation(self, x):
        return 1 if x >= 0 else 0

    def predict(self, inputs):
        weighted_sum = np.dot(inputs, self.weights) + self.bias
        return self.activation(weighted_sum)

    def fit(self, X, y):
        for epoch in range(self.epochs):
            total_error = 0
            for inputs, label in zip(X, y):
                prediction = self.predict(inputs)
                error = label - prediction
                self.weights += self.learning_rate * error * inputs
                self.bias += self.learning_rate * error
                total_error += abs(error)
            if total_error == 0:
                break 

    def evaluate(self, X, y):
        correct_predictions = 0
        for inputs, label in zip(X, y):
            if self.predict(inputs) == label:
                correct_predictions += 1
        accuracy = correct_predictions / len(y)
        return accuracy
    
# NAND Dataset
nand_input = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
nand_output = np.array([1, 1, 1, 0])  

# XOR Dataset
xor_input = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
xor_output = np.array([0, 1, 1, 0])  

# Train Perceptron for NAND
print("Training Perceptron for NAND Function")
nand_perceptron = Perceptron(input_size=2, learning_rate=0.1, epochs=100)
nand_perceptron.fit(nand_input, nand_output)
nand_accuracy = nand_perceptron.evaluate(nand_input, nand_output)
print(f"NAND Accuracy: {nand_accuracy * 100:.2f}%")

# Train Perceptron for XOR
print("\nTraining Perceptron for XOR Function")
xor_perceptron = Perceptron(input_size=2, learning_rate=0.1, epochs=100)
xor_perceptron.fit(xor_input, xor_output)
xor_accuracy = xor_perceptron.evaluate(xor_input, xor_output)
print(f"XOR Accuracy: {xor_accuracy * 100:.2f}%")

## Code Explanation

### **Perceptron Class**
- **Initializes** small random weights and a bias term.
- Uses a **step activation function**:  
  - Outputs `1` if the weighted sum is **≥ 0**, otherwise `0`.
- **Trains using weight updates**:  
  - `weight = weight + learning_rate * (target - prediction) * input`
  - `bias = bias + learning_rate * (target - prediction)`
- **Evaluates model accuracy** by comparing predictions with actual values.

### **Training on NAND Gate**
- Provides input-output pairs for the **NAND logic gate**.
- The perceptron **successfully** learns and achieves **~100% accuracy**.

### **Training on XOR Gate**
- Provides input-output pairs for the **XOR logic gate**.
- The perceptron **fails** since **XOR is not linearly separable**.

---
```
## Results and Performance

| Gate  | Accuracy (%) | Predictions  | Expected Output | Linearly Separable? |
|--------|------------|-------------|----------------|----------------------|
| **NAND** | ~100.00%  | `[1, 1, 1, 0]`  | `[1, 1, 1, 0]`  | ✅ Yes  |
| **XOR**  | ~50.00%   | `[0, 0, 0, 0]` (or incorrect) | `[0, 1, 1, 0]`  | ❌ No  |

---

## Limitations
- A **single-layer perceptron** can only learn **linearly separable** functions.
- **XOR cannot be learned** without adding **hidden layers** (MLP required).

