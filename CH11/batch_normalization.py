###
# Implementation of batch normalization on MNIST
###
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#
# Construction Phase
#
# Define shape of network
n_inputs = 28 * 28  # MNIST images are 28x28 greyscale
n_hidden1 = 300     # 300 neurons in hidden layer 1
n_hidden2 = 200     # 100 neurons in hidden layer 2
n_outputs = 10      # 10-classification problem

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")  # X is a matrix of mini-batch examples
y = tf.placeholder(tf.int64, shape=(None), name="y")
training = tf.placeholder_with_default(False, shape=(), name='training')  # Boolean flag used when computing batch level statistics

# Define DNN architecture
with tf.name_scope("dnn"):
	hidden1 = tf.layers.dense(X, n_hidden1, name="hidden1")
	bn1 = tf.layers.batch_normalization(hidden1, training=training, momentum=0.9)
	bn1_activation = tf.nn.elu(bn1)

	hidden2 = tf.layers.dense(bn1_activation, n_hidden2, name="hidden2", activation=tf.nn.relu)
	bn2 = tf.layers.batch_normalization(hidden2, training=training, momentum=0.9)
	bn2_activation = tf.nn.elu(bn2)
	
	logits_unnormalized = tf.layers.dense(bn2_activation, n_outputs, name="outputs")
	logits = tf.layers.batch_normalization(logits_unnormalized, training=training, momentum=0.9)

# Define mean cross entropy loss
with tf.name_scope("loss"):
	xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
	loss = tf.reduce_mean(xentropy, name="loss")

# Define optimizer
learning_rate = 0.1

with tf.name_scope("train"):
	optimizer = tf.train.GradientDescentOptimizer(learning_rate)
	training_op = optimizer.minimize(loss)

# Define evaluation metric
with tf.name_scope("eval"):
	correct = tf.nn.in_top_k(logits, y, 1)
	accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

# Define initializer and saver nodes
init = tf.global_variables_initializer()

#
# Execution Phase
#
mnist = input_data.read_data_sets("/tmp/data/")
n_epochs = 30
batch_size = 50
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

with tf.Session() as sess:
	init.run()

	for epoch in range(n_epochs):
		for iteration in range(mnist.train.num_examples // batch_size):
			# Get batch data and train
			X_batch, y_batch = mnist.train.next_batch(batch_size)
			sess.run([training_op, update_ops], feed_dict={training: True, X: X_batch, y: y_batch})

		# Evaluate model performance
		acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
		acc_test = accuracy.eval(feed_dict={X: mnist.test.images, y:mnist.test.labels})
		print(epoch, "Train accuracy: ", acc_train, "Test accuracy: ", acc_test)