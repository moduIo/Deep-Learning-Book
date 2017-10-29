######
# Implements gradient descent on Housing dataset using MSE objective function
######
import tensorflow as tf
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler

# Get dataset
dataset = fetch_california_housing()
m, n = dataset.data.shape

# Concatenate a bias constant vector b to the X data
# X' <- [b X]
X_biased = np.c_[np.ones((m, 1)), dataset.data]

# Scale X' to normalize the features
X_scaled = StandardScaler().fit_transform(X_biased)

# Reshape target values to be a vector in R^{m x 1}
y_vector = dataset.target.reshape(-1, 1)

# Define gradient descent parameters
n_epochs = 1000
eta = 0.01  # Learning rate

# Define tf nodes
X = tf.constant(X_scaled, dtype=tf.float32, name="X")
y = tf.constant(y_vector, dtype=tf.float32, name="y")
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0), name="theta")  # Init theta to random values in [-1, 1]
y_pred = tf.matmul(X, theta, name="predictions")
mse = tf.reduce_mean(tf.square(y_pred - y), name="mse")  # MSE is mean((y' - y)^2)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=eta)
training_op = optimizer.minimize(mse)

# Run the graph
init = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)

	# Print the MSE every 100 iterations
	for epoch in range(n_epochs):
		if epoch % 100 == 0:
			print "Epoch " + str(epoch) + " | MSE = " + str(mse.eval())
		sess.run(training_op)

	best_theta = theta.eval()