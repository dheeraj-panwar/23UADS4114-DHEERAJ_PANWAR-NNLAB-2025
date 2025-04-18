import tensorflow.compat.v1 as tf
import numpy as np
import tensorflow_datasets as tfds

# Disable TensorFlow 2 behavior
tf.disable_v2_behavior()

# Load MNIST dataset
mnist = tfds.load('mnist', split=['train', 'test'], as_supervised=True)
train_data, test_data = mnist

# Network Parameters
input_size = 784  # 28x28 images
hidden_layer1_size = 128
hidden_layer2_size = 64
output_size = 10
learning_rate = 0.01
epochs = 10
batch_size = 100

# Function to preprocess images
def preprocess(images, labels):
    images = tf.cast(images, tf.float32) / 255.0  # Convert to float32 and normalize
    images = tf.reshape(images, [784])  # Ensure it's (784,) instead of (1, 784)
    labels = tf.one_hot(labels, depth=10)  # One-hot encode labels
    return images, labels

# Apply preprocessing and batch the data
train_data = train_data.map(preprocess).batch(batch_size)
test_data = test_data.map(preprocess).batch(batch_size)

# Define placeholders
X = tf.placeholder(tf.float32, [None, input_size])
Y = tf.placeholder(tf.float32, [None, output_size])

# Initialize weights and biases
weights = {
    'h1': tf.Variable(tf.random_normal([input_size, hidden_layer1_size])),
    'h2': tf.Variable(tf.random_normal([hidden_layer1_size, hidden_layer2_size])),
    'out': tf.Variable(tf.random_normal([hidden_layer2_size, output_size]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([hidden_layer1_size])),
    'b2': tf.Variable(tf.random_normal([hidden_layer2_size])),
    'out': tf.Variable(tf.random_normal([output_size]))
}

# Define the neural network
def neural_network(x):
    layer1 = tf.nn.relu(tf.add(tf.matmul(x, weights['h1']), biases['b1']))
    layer2 = tf.nn.relu(tf.add(tf.matmul(layer1, weights['h2']), biases['b2']))
    output_layer = tf.add(tf.matmul(layer2, weights['out']), biases['out'])
    return output_layer

# Compute predictions
logits = neural_network(X)
predictions = tf.nn.softmax(logits)

# Define loss and optimizer
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# Define accuracy metric
correct_pred = tf.equal(tf.argmax(predictions, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Train the model
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(epochs):
        avg_loss = 0
        total_batches = 0

        iterator = tf.compat.v1.data.make_one_shot_iterator(train_data)
        next_batch = iterator.get_next()

        while True:
            try:
                batch_x, batch_y = sess.run(next_batch)
                _, c = sess.run([optimizer, loss], feed_dict={X: batch_x, Y: batch_y})
                avg_loss += c
                total_batches += 1
            except tf.errors.OutOfRangeError:
                break  # End of dataset

        avg_loss /= total_batches
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

    # Evaluate model
    test_acc = []
    iterator = tf.compat.v1.data.make_one_shot_iterator(test_data)
    next_batch = iterator.get_next()

    while True:
        try:
            batch_x, batch_y = sess.run(next_batch)
            acc = sess.run(accuracy, feed_dict={X: batch_x, Y: batch_y})
            test_acc.append(acc)
        except tf.errors.OutOfRangeError:
            break

    print(f"Test Accuracy: {np.mean(test_acc):.4f}")