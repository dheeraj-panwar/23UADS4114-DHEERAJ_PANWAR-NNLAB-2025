import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, regularizers
from tensorflow.keras.optimizers import Adam, SGD

# Load Fashion MNIST dataset
(x_train, y_train), (x_test, y_test) = datasets.fashion_mnist.load_data()

# Normalize the data
x_train = x_train / 255.0
x_test = x_test / 255.0

# Reshape for CNN input
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

def create_model(filter_size=3, regularization=None, optimizer='adam'):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (filter_size, filter_size), activation='relu', input_shape=(28,28,1),
                            kernel_regularizer=regularization))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(64, (filter_size, filter_size), activation='relu',
                            kernel_regularizer=regularization))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu', kernel_regularizer=regularization))
    model.add(layers.Dense(10, activation='softmax'))

    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Function to train and evaluate
results = {}
def train_and_evaluate(filter_size=3, regularization=None, batch_size=64, optimizer='adam', label='default'):
    model = create_model(filter_size=filter_size, regularization=regularization, optimizer=optimizer)
    history = model.fit(x_train, y_train, epochs=5, batch_size=batch_size, validation_split=0.1, verbose=0)
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    results[label] = {'history': history, 'test_accuracy': test_acc}
    print(f"{label}: Test Accuracy = {test_acc:.4f}")

# 1. Effect of filter size
for fs in [3, 5, 7]:
    train_and_evaluate(filter_size=fs, label=f'Filter size {fs}')

# 2. Effect of regularization
for reg in [None, regularizers.l2(0.001)]:
    label = 'No regularization' if reg is None else 'L2 regularization'
    train_and_evaluate(regularization=reg, label=label)

# 3. Effect of batch size
for bs in [32, 64, 128]:
    train_and_evaluate(batch_size=bs, label=f'Batch size {bs}')

# 4. Effect of optimizer
for opt in ['adam', 'sgd']:
    train_and_evaluate(optimizer=opt, label=f'Optimizer {opt}')

# Plotting Results
labels = list(results.keys())
accuracies = [results[k]['test_accuracy'] for k in labels]

plt.figure(figsize=(12,6))
plt.barh(labels, accuracies, color='skyblue')
plt.xlabel('Test Accuracy')
plt.title('Comparison of Different Parameters on Fashion MNIST')
plt.grid(True)
plt.show()
