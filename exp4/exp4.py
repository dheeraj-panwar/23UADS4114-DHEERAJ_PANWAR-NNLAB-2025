import tensorflow.compat.v1 as tf
import numpy as np
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import time
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Disable TensorFlow 2 behavior
tf.disable_v2_behavior()

# Load MNIST dataset
mnist = tfds.load('mnist', split=['train', 'test'], as_supervised=True)
train_data, test_data = mnist

# Network Parameters
input_size = 784  # 28x28 images
output_size = 10
learning_rate = 0.01
epochs = 50
batch_size = 10

# Hyperparameters to tune
activation_functions = {'relu': tf.nn.relu, 'sigmoid': tf.nn.sigmoid, 'tanh': tf.nn.tanh}
hidden_layer_sizes = [256, 128, 64]

# Function to preprocess images
def preprocess(images, labels):
    images = tf.cast(images, tf.float32) / 255.0  # Normalize
    images = tf.reshape(images, [784])  # Flatten
    labels = tf.one_hot(labels, depth=10)  # One-hot encode labels
    return images, labels

# Apply preprocessing and batch the data
train_data = train_data.map(preprocess).batch(batch_size)
test_data = test_data.map(preprocess).batch(batch_size)

# Placeholder for results
results = []

# Iterate over hyperparameters
for activation_name, activation_func in activation_functions.items():
    for hidden_layer_size in hidden_layer_sizes:
        print(f"Training with Activation: {activation_name}, Hidden Layer Size: {hidden_layer_size}")
        
        # Define placeholders
        X = tf.placeholder(tf.float32, [None, input_size])
        Y = tf.placeholder(tf.float32, [None, output_size])
        
        # Initialize weights and biases
        weights = {
            'h1': tf.Variable(tf.random_normal([input_size, hidden_layer_size])),
            'out': tf.Variable(tf.random_normal([hidden_layer_size, output_size]))
        }
        biases = {
            'b1': tf.Variable(tf.random_normal([hidden_layer_size])),
            'out': tf.Variable(tf.random_normal([output_size]))
        }
        
        # Define the neural network
        def neural_network(x):
            layer1 = activation_func(tf.add(tf.matmul(x, weights['h1']), biases['b1']))
            output_layer = tf.add(tf.matmul(layer1, weights['out']), biases['out'])
            return output_layer
        
        logits = neural_network(X)
        predictions = tf.nn.softmax(logits)
        
        # Define loss and optimizer
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y))
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
        
        # Define accuracy metric
        correct_pred = tf.equal(tf.argmax(predictions, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        
        # Start session
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            train_losses, train_accuracies = [], []
            start_time = time.time()
            
            for epoch in range(epochs):
                avg_loss = 0
                total_batches = 0
                iterator = tf.compat.v1.data.make_one_shot_iterator(train_data)
                next_batch = iterator.get_next()
                
                while True:
                    try:
                        batch_x, batch_y = sess.run(next_batch)
                        _, c, acc = sess.run([optimizer, loss, accuracy], feed_dict={X: batch_x, Y: batch_y})
                        avg_loss += c
                        total_batches += 1
                    except tf.errors.OutOfRangeError:
                        break
                
                avg_loss /= total_batches
                train_losses.append(avg_loss)
                train_accuracies.append(acc)
                print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}, Accuracy: {acc:.4f}")
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Evaluate model
            test_acc = []
            y_true, y_pred = [], []
            iterator = tf.compat.v1.data.make_one_shot_iterator(test_data)
            next_batch = iterator.get_next()
            
            while True:
                try:
                    batch_x, batch_y = sess.run(next_batch)
                    acc, preds = sess.run([accuracy, predictions], feed_dict={X: batch_x, Y: batch_y})
                    test_acc.append(acc)
                    y_true.extend(np.argmax(batch_y, axis=1))
                    y_pred.extend(np.argmax(preds, axis=1))
                except tf.errors.OutOfRangeError:
                    break
            
            final_test_accuracy = np.mean(test_acc)
            results.append((activation_name, hidden_layer_size, final_test_accuracy, execution_time))
            
            print(f"Final Test Accuracy: {final_test_accuracy:.4f}")
            print(f"Execution Time: {execution_time:.2f} seconds\n")
            
            # Plot loss and accuracy curves
            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            plt.plot(range(epochs), train_losses, label='Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.title(f'Loss Curve - {activation_name}, {hidden_layer_size}')
            plt.legend()
            
            plt.subplot(1, 2, 2)
            plt.plot(range(epochs), train_accuracies, label='Accuracy', color='green')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            plt.title(f'Accuracy Curve - {activation_name}, {hidden_layer_size}')
            plt.legend()
            plt.show()
            
            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title(f'Confusion Matrix - {activation_name}, {hidden_layer_size}')
            plt.show()

# Print summary of results
print("\nHyperparameter Tuning Results:")
for activation, hl_size, acc, time_taken in results:
    print(f"Activation: {activation}, Hidden Layer: {hl_size}, Test Accuracy: {acc:.4f}, Time: {time_taken:.2f}s")