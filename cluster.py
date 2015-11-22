"""
TFKMeansCluster function made by Sachin Joglekar:
https://codesachin.wordpress.com/2015/11/14/k-means-clustering-with-tensorflow/

"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

from random import choice, shuffle
from numpy import array


def TFKMeansCluster(vectors, noofclusters):
	"""
	K-Means Clustering using TensorFlow.

	'vectors' should be a n*k 2-D NumPy array, where n is the number
	of vectors of dimensionality k.
	'noofclusters' should be an integer.
	"""

	noofclusters = int(noofclusters)
	assert noofclusters < len(vectors)

	## Find out the dimensionality
	dim = len(vectors[0])

	## Will help select random centroids from among the available vectors
	vector_indices = list(range(len(vectors)))
	shuffle(vector_indices)

	## GRAPH OF COMPUTATION
	## We initialize a new graph and set it as the default during each run
	## of this algorithm. This ensures that as this function is called
	## multiple times, the default graph doesn't keep getting crowded with
	## unused ops and Variables from previous function calls.

	graph = tf.Graph()

	with graph.as_default():
		## SESSION OF COMPUTATION

		sess = tf.Session()

		### CONSTRUCTING THE ELEMENTS OF COMPUTATION

		### First lets ensure we have a Variable vector for each centroid,
		### initialized to one of the vectors from the available data points
		centroids = [tf.Variable((vectors[vector_indices[i]]))
					 for i in range(noofclusters)]
		### These nodes will assign the centroid Variables the appropriate
		### values
		centroid_value = tf.placeholder("float64", [dim])
		cent_assigns = []
		for centroid in centroids:
			cent_assigns.append(tf.assign(centroid, centroid_value))

		### Variables for cluster assignments of individual vectors
		### (initialized to 0 at first)
		assignments = [tf.Variable(0) for i in range(len(vectors))]
		### These nodes will assign an assignment Variable the appropriate
		### value
		assignment_value = tf.placeholder("int32")
		cluster_assigns = []
		for assignment in assignments:
			cluster_assigns.append(tf.assign(assignment,
				assignment_value))

		### Now lets construct the node that will compute the mean
		## The placeholder for the input
		mean_input = tf.placeholder("float", [None, dim])
		## The Node/op takes the input and computes a mean along the 0th
		## dimension, i.e. the list of input vectors
		mean_op = tf.reduce_mean(mean_input, 0)

		### Node for computing Euclidean distances
		## Placeholders for input
		v1 = tf.placeholder("float", [dim])
		v2 = tf.placeholder("float", [dim])
		euclid_dist = tf.sqrt(tf.reduce_sum(tf.pow(tf.sub(
			v1, v2), 2)))

		### This node will figure out which cluster to assign a vector to,
		### based on Euclidean distances of the vector from the centroids.
		## Placeholder for input
		centroid_distances = tf.placeholder("float", [noofclusters])
		cluster_assignment = tf.argmin(centroid_distances, 0)

		### INITIALIZING STATE VARIABLES

		### This will help initialization of all Variables defined with
		### respect to the graph. The Variable-initializer should be defined
		### after all the Variables have been constructed, so that each of
		### them will be included in the initialization.
		init_op = tf.initialize_all_variables()

		## Initialize all variables
		sess.run(init_op)

		### CLUSTERING ITERATIONS

		## Now perform the Expectation-Maximization steps of K-Means
		## clustering iterations. To keep things simple, we will only do a
		## set number of iterations, instead of using a Stopping Criterion.
		noofiterations = 50
		for iteration_n in range(noofiterations):

			### EXPECTATION STEP
			### Based on the centroid locations till last iteration, compute
			### the _expected_ centroid assignments.
			## Iterate over each vector
			for vector_n in range(len(vectors)):
				vect = vectors[vector_n]
				## Compute Euclidean distance between this vector and each
				## centroid. Remember that this list cannot be named
				## 'centroid_distances', since that is the input to the
				## cluster assignment node.
				distances = [sess.run(euclid_dist, feed_dict={
					v1: vect, v2: sess.run(centroid)})
							 for centroid in centroids]
				## Now use the cluster assignment node, with the distances
				## as the input
				assignment = sess.run(cluster_assignment, feed_dict = {
					centroid_distances: distances})
				## Now assign the value to the appropriate state variable
				sess.run(cluster_assigns[vector_n], feed_dict={
					assignment_value: assignment})

			### MAXIMIZATION STEP
			## Based on the expected state computed from the Expectation
			## Step, compute the locations of the centroids so as to
			## maximize the overall objective of minimizing within-cluster
			## Sum-of-Squares
			for cluster_n in range(noofclusters):
				## Collect all the vectors assigned to this cluster
				assigned_vects = [vectors[i] for i in range(len(vectors))
								  if sess.run(assignments[i]) == cluster_n]
				## Compute new centroid location
				new_location = sess.run(mean_op, feed_dict={
					mean_input: array(assigned_vects)})
				## Assign value to appropriate variable
				sess.run(cent_assigns[cluster_n], feed_dict={
					centroid_value: new_location})

		## Return centroids and assignments
		centroids = sess.run(centroids)
		assignments = sess.run(assignments)
		return centroids, assignments


def generateRandomData(name):
	"""Generate random data from 2D gaussian distributions with different means."""
	samples = 20
	cov = np.array([[21,8], [8,21]])
	d1 = np.random.multivariate_normal((22,22), cov, samples)
	d2 = np.random.multivariate_normal((33,33), cov, samples)
	d3 = np.random.multivariate_normal((33,22), cov, samples)
	dataraw = pd.DataFrame(np.concatenate((d1,d2,d3)), columns=['x','y'])
	## TODO include column to distinguish the data from each distribution
	dataraw.to_csv(name)
	return d1, d2, d3, dataraw



if __name__ == '__main__':
	DATAFILE = 'testdata.csv'
	# DATAFILE = 'points.csv'

	## Reads data from a file
	# dataraw = pd.read_csv(DATAFILE)

	## Generate data from a few gaussian distributions
	d1, d2, d3, dataraw = generateRandomData(DATAFILE)

	## Create an unsorted matrix of data of float type
	data = dataraw.get_values().astype(np.float, copy=True)
	np.random.shuffle(data)
	## Defines the number of clusters and the color for each one
	k = 3
	colorByK = ['b', 'r', 'y', 'g', 'm', 'c']	# 6 clusters max !

	## Show and plot the original data (with the good clusters colored)
	print dataraw
	plt.plot(d1[:,0], d1[:,1], colorByK[0]+'x')
	plt.plot(d2[:,0], d2[:,1], colorByK[1]+'x')
	plt.plot(d3[:,0], d3[:,1], colorByK[2]+'x')
	# dataraw.plot.scatter('x','y',style='k.')

	## Compute the centroids and assignments with K-Means
	centroids, assignments = TFKMeansCluster(data, k)
	print pd.DataFrame(centroids, columns=['x','y'])
	print assignments

	## Plot the clustering result
	for i in range(k):
		pts = data[filter(lambda x: assignments[x]==i, range(len(data)))]
		cent = centroids[i]
		plt.plot(cent[0], cent[1], colorByK[i]+'o')
		plt.plot(pts[:,0], pts[:,1], colorByK[i]+'.')

	plt.show()

	## TODO count points incorrectly assigned
