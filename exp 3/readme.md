# MNIST Handwritten Digit Classification using TensorFlow

## Objective
This project aims to develop a deep learning model using TensorFlow to classify handwritten digits from the MNIST dataset. The model is a fully connected neural network (Multi-Layer Perceptron) that learns to recognize digits (0-9) based on their pixel values.

## Description of the Model
The model is a **three-layer feed-forward neural network** implemented using TensorFlow (without Keras). The architecture consists of:

- **Input Layer:** 784 neurons (28x28 flattened grayscale image pixels)
- **Hidden Layer 1:** 128 neurons with ReLU activation
- **Hidden Layer 2:** 64 neurons with ReLU activation
- **Output Layer:** 10 neurons (one for each digit class) with softmax activation

The model is trained using **cross-entropy loss** and optimized with the **Adam optimizer**. The training is performed for **10 epochs** with a batch size of **100**.

## Description of the Code
The implementation follows these key steps:

1. **Import Libraries:** Uses `tensorflow.compat.v1`, `numpy`, and `tensorflow_datasets`.
2. **Disable TensorFlow 2 Behavior:** Ensures TensorFlow 1.x compatibility.
3. **Load & Preprocess Data:** 
   - Downloads MNIST dataset.  
   - Normalizes pixel values to `[0,1]`.  
   - Converts labels to one-hot encoding.  
4. **Define Neural Network:**  
   - Three-layer MLP with ReLU activations.  
   - Softmax for classification.  
5. **Train Model:**  
   - Uses cross-entropy loss and Adam optimizer.  
   - Trains for **10 epochs** with batch size **100**.  
6. **Evaluate Performance:**  
   - Computes test accuracy after training.  
   - Achieves **93.11% accuracy** on MNIST test data.  

## Performance Evaluation
After running the model for **10 epochs**, the performance was:

### **Training Loss per Epoch:**

| Epoch | Loss   |
|-------|--------|
| 1     | 11.8013 |
| 2     | 1.1991  |
| 3     | 0.8189  |
| 4     | 0.6115  |
| 5     | 0.4783  |
| 6     | 0.3764  |
| 7     | 0.3206  |
| 8     | 0.2887  |
| 9     | 0.2502  |
| 10    | 0.2312  |

### **Test Accuracy:**

The model successfully achieves an accuracy of **93.11%** on the MNIST test dataset, demonstrating its effectiveness in digit classification.


