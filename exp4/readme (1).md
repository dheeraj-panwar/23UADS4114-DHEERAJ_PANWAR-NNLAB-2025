## **Objective**
WAP to evaluate the performance of implemented three-layer neural network with variations in activation functions, size of hidden layer, learning rate, batch size and number of epochs.

## **Model Description**
This project trains a **Feedforward Neural Network (FNN)** on the **MNIST dataset** using TensorFlow.  
The model explores different **activation functions** and **hidden layer sizes** to analyze their impact on performance.

The model consists of:
- **Input Layer**: 784 neurons (flattened 28x28 images)
- **Hidden Layer**: Single layer with tunable sizes (**256, 128, 64 neurons**)
- **Output Layer**: 10 neurons (digits 0-9)
- **Activation Functions**: ReLU, Sigmoid, Tanh
- **Loss Function**: Softmax Cross-Entropy
- **Optimizer**: Adam Optimizer

## **Description of Code**
### **1. Data Preprocessing**
- Loads **MNIST dataset** using `tensorflow_datasets`
- Normalizes pixel values (0-255 → 0-1)
- Converts labels to **one-hot encoding**
- Flattens images from **28×28 → 784**

### **2. Neural Network Construction**
- Uses a **single hidden layer** with:
  - Variable **number of neurons** (256, 128, 64)
  - Different **activation functions** (ReLU, Sigmoid, Tanh)
- Output layer applies **Softmax activation** for classification

### **3. Training & Evaluation**
- **Trains for 50 epochs**
- Computes:
  - **Loss curve**
  - **Accuracy curve**
  - **Confusion matrix**
- Evaluates final **test accuracy** and **execution time**

### **4. Results Export**
- Saves results in **Excel file** 
- Stores :-
  - Activation Function  
  - Hidden Layer Size  
  - Loss Curve  
  - Accuracy Curve  
  - Confusion Matrix  
  - Test Accuracy  
  - Execution Time  

## **Performance Evaluation**
Evaluated using accuracy on the test set and execution time for each configuration.
Results are printed at the end summarizing:
- Activation function
- Hidden layer size
- Final test accuracy
- Training and evaluation time
- Confusion matrix provides insight into per-class performance and misclassifications.
- Loss and accuracy plots show learning behavior across epochs.

## **My Comments**
- **ReLU performed best** with **higher hidden layer sizes**.
- **Sigmoid had slower convergence** and lower accuracy.
- **Tanh performed better than Sigmoid** but **worse than ReLU**.
- Execution time **varied based on activation function & hidden layer size**.



