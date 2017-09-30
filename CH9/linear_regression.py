###
# Implementation of linear regression on the California Housing dataset.
###
import tensorflow as tf
import numpy as np
from sklearn.datasets import fetch_california_housing

# Get dataset
dataset = fetch_california_housing()
m, n = dataset.data.shape

# Concatenate a bias constant vector b to the X data
# X' <- [b X]
X_biased = np.c_[np.ones((m, 1)), dataset.data]

# Reshape target values to be a vector in R^{m x 1}
y_vector = dataset.target.reshape(-1, 1)

# Define graph
X = tf.constant(X_biased, dtype=tf.float32, name="X")
y = tf.constant(y_vector, dtype=tf.float32, name="y")
XT = tf.transpose(X)

# The solution is the familiar OLS: w = (XT * X)^{-1} * XT * y
w = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT, X)), XT), y)

# Evaluate graph
with tf.Session() as sess:
	w_val = w.eval()

print w_val