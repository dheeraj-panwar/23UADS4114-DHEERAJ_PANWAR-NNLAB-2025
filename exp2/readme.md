#### Objective
This project implements the Perceptron Learning Algorithm using NumPy in Python. The goal is to evaluate the performance of a single perceptron on the NAND and XOR truth tables and then extend it to a Multi-Layer Perceptron (MLP) to classify Boolean functions.

#### Description of the Model
This implementation follows a multi-layer perceptron (MLP) approach where:
- Individual perceptrons are trained to recognize Boolean functions (NAND, Custom functions).
- The outputs of these perceptrons are then combined to train a final perceptron for binary classification.
- The model is evaluated using accuracy.

#### Python Implementation
```python
import numpy as np

class Perceptron:
    def __init__(self, input_size, learning_rate=0.1, epochs=100):
        self.weights = np.random.randn(input_size + 1)  # +1 for bias
        self.learning_rate = learning_rate
        self.epochs = epochs

    def activation(self, x):
        return 1 if x >= 0 else 0

    def predict(self, x):
        x = np.insert(x, 0, 1)  # Add bias term
        return self.activation(np.dot(self.weights, x))

    def train(self, X, y):
        X = np.c_[np.ones(X.shape[0]), X]  # Add bias column
        for _ in range(self.epochs):
            for i in range(X.shape[0]):
                y_pred = self.activation(np.dot(self.weights, X[i]))
                self.weights += self.learning_rate * (y[i] - y_pred) * X[i]

    def evaluate(self, X, y):
        y_pred = np.array([self.predict(x) for x in X])
        accuracy = np.mean(y_pred == y) * 100
        return accuracy, y_pred

def train_perceptron(X, y, name):
    p = Perceptron(input_size=X.shape[1])
    p.train(X, y)
    accuracy, predictions = p.evaluate(X, y)
    print(f"{name} Accuracy: {accuracy:.2f}% | Predictions: {predictions}")
    return predictions, y

fun_X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

fun1_y = np.array([0, 0, 0, 1])  
fun2_y = np.array([0, 0, 1, 0])  
fun3_y = np.array([0, 1, 0, 0])  
fun4_y = np.array([1, 0, 0, 0]) 

fun1_predictions, _ = train_perceptron(fun_X, fun1_y, "Fun1")
fun2_predictions, _ = train_perceptron(fun_X, fun2_y, "Fun2")
fun3_predictions, _ = train_perceptron(fun_X, fun3_y, "Fun3")
fun4_predictions, _ = train_perceptron(fun_X, fun4_y, "Fun4")

final_X = np.column_stack([fun1_predictions, fun2_predictions, fun3_predictions, fun4_predictions])
final_y = np.array([0, 1, 1, 0])

final_predictions, actual_y = train_perceptron(final_X, final_y, "Final Perceptron")
```

#### Explanation of the Code
1. **Libraries Used**
   - NumPy is used for numerical operations and matrix calculations.
2. **Perceptron Class**
   - The perceptron model initializes random weights and applies a learning rule based on error correction.
3. **Activation Function**
   - Uses a threshold-based step function to classify inputs.
4. **Training Process**
   - Implements the perceptron learning algorithm with weight updates based on errors.
5. **Evaluation**
   - Accuracy is computed based on correct classifications.

#### Results and Performance
- Fun1 Accuracy: **100.00%** | Predictions: `[0 0 0 1]`
- Fun2 Accuracy: **100.00%** | Predictions: `[0 0 1 0]`
- Fun3 Accuracy: **100.00%** | Predictions: `[0 1 0 0]`
- Fun4 Accuracy: **100.00%** | Predictions: `[1 0 0 0]`
- Final Perceptron Accuracy: **100.00%** | Predictions: `[0 1 1 0]`


