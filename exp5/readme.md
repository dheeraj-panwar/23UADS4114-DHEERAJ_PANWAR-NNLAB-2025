## Objective

To train and evaluate a Convolutional Neural Network (CNN) using the Keras library to classify the Fashion MNIST dataset. The objective is to analyze the effect of different hyperparameters, such as filter size, regularization, batch size, and optimization algorithm, on model performance.

---

## Model Description

A Convolutional Neural Network (CNN) is used for image classification with the following architecture:

- **Input Layer**: 28x28 grayscale images reshaped to (28, 28, 1)
- **Conv Layer 1**: 32 filters, variable kernel size, ReLU activation
- **MaxPooling 1**: 2x2 pool size
- **Conv Layer 2**: 64 filters, same kernel size, ReLU activation
- **MaxPooling 2**: 2x2 pool size
- **Flatten Layer**
- **Dense Layer**: 64 neurons, ReLU activation
- **Output Layer**: 10 neurons, Softmax activation
---

## Code Description

- **Data Loading & Preprocessing**:
  - Loads Fashion MNIST dataset
  - Normalizes pixel values to `[0, 1]`
  - Reshapes data for CNN input

- **Model Creation**:
  - Accepts `filter_size`, `regularization`, and `optimizer` as arguments
  - Builds a sequential CNN model accordingly

- **Training & Evaluation**:
  - `train_and_evaluate()` function trains the model and evaluates its test accuracy
  - Results are saved for each configuration

- **Hyperparameter Experiments**:
  - **Filter Size**: 3, 5, 7
  - **Regularization**: None, L2 (0.001)
  - **Batch Size**: 32, 64, 128
  - **Optimizers**: Adam, SGD

- **Visualization**:
  - Horizontal bar chart is plotted to compare the test accuracies of all configurations

---

## Performance Evaluation

| Parameter           | Setting              | Test Accuracy |
|---------------------|----------------------|----------------|
| **Filter Size**     | 3                    | **0.9002**     |
|                     | 5                    | 0.8897         |
|                     | 7                    | 0.8716         |
| **Regularization**  | None                 | **0.9025**     |
|                     | L2 (0.001)           | 0.8789         |
| **Batch Size**      | 32                   | 0.8908         |
|                     | 64                   | **0.8995**     |
|                     | 128                  | 0.8954         |
| **Optimizer**       | Adam                 | **0.8990**     |
|                     | SGD                  | 0.8317         |

---

## My Comment

- Found that **smaller filter sizes (3x3)** perform better on the Fashion MNIST dataset.
- **Adam optimizer** significantly outperformed **SGD** in terms of accuracy and convergence speed.
- **Batch size 64** gave a good balance between training time and performance.
- **Regularization (L2)** slightly reduced the performance, indicating potential underfitting.
---
