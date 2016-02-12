import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf


## Implementacion de KMeans en TensorFlow
def TFKMeans(points, noofclusters, noofiterations):
	noofclusters = int(noofclusters)
	assert noofclusters < len(points)
	assert noofiterations > 0
	dim = len(points[0])

	indices = range(len(points))
	np.random.shuffle(indices)

	## GRAPH OF COMPUTATION
	## We initialize a new graph and set it as the default during each run
	## of this algorithm. This ensures that as this function is called
	## multiple times, the default graph doesn't keep getting crowded with
	## unused ops and Variables from previous function calls.
	graph = tf.Graph()
	with graph.as_default():
		sess = tf.Session()

		## First lets ensure we have a Variable for each centroid,
		## initialized to one of the points from the available data points
		centroids = [tf.Variable((points[indices[i]])) for i in range(noofclusters)]

		## These nodes will assign the centroid Variables the appropriate values
		centroid_value = tf.placeholder("float64", [dim])
		cent_assigns = []
		for centroid in centroids:
			cent_assigns.append(tf.assign(centroid, centroid_value))

		## Variables for cluster assignments of individual points
		assignments = [tf.Variable(0) for i in range(len(points))]
		## These nodes will assign an assignment Variable the appropriate value
		assignment_value = tf.placeholder("int32")
		cluster_assigns = []
		for assignment in assignments:
			cluster_assigns.append(tf.assign(assignment, assignment_value))

		## Now lets construct the node that will compute the mean
		mean_input = tf.placeholder("float", [None, dim])
		## The Node/op takes the input and computes a mean along the
		## 0th dimension, i.e. the list of input points
		mean_op = tf.reduce_mean(mean_input, 0)

		v1 = tf.placeholder("float", [dim])
		v2 = tf.placeholder("float", [dim])
		euclid_dist = tf.sqrt(tf.reduce_sum(tf.pow(tf.sub(v1, v2), 2)))

		## This node will figure out which cluster to assign a point to,
		## based on Euclidean distances of the point from the centroids.
		centroid_distances = tf.placeholder("float", [noofclusters])
		cluster_assignment = tf.argmin(centroid_distances, 0)

		## INITIALIZING STATE VARIABLES
		## This will help initialization of all Variables defined with
		## respect to the graph. The Variable-initializer should be defined
		## after all the Variables have been constructed, so that each of
		## them will be included in the initialization.
		init_op = tf.initialize_all_variables()
		sess.run(init_op)

		for iteration_n in range(noofiterations):
			## EXPECTATION STEP
			for pt_n in range(len(points)):
				p = points[pt_n]
				distances = [sess.run(euclid_dist, feed_dict={
					v1: p, v2: sess.run(centroid)}) for centroid in centroids]
				assignment = sess.run(cluster_assignment, feed_dict = {
					centroid_distances: distances})
				sess.run(cluster_assigns[pt_n], feed_dict={assignment_value: assignment})

			## MAXIMIZATION STEP
			for cluster_n in range(noofclusters):
				assigned_pts = [points[i] for i in range(len(points))
								if sess.run(assignments[i]) == cluster_n]
				new_location = sess.run(mean_op, feed_dict={mean_input: np.array(assigned_pts)})
				sess.run(cent_assigns[cluster_n], feed_dict={centroid_value: new_location})

		centroids = sess.run(centroids)
		assignments = sess.run(assignments)
		writer = tf.python.training.summary_io.SummaryWriter('/Users/sergio/Desktop/tensorflow/')
		writer.add_graph(sess.graph_def)
		writer.close()
		sess.close()
		return centroids, assignments


##################

## Generacion de datos aleatorios
samples = 20
cov = np.array([[12,4], [4,14]])
group1 = np.random.multivariate_normal((37,33), cov, samples)
group2 = np.random.multivariate_normal((19,21), cov, samples)

data = pd.DataFrame(np.concatenate((group1, group2)), columns=['x','y'])
points = data.get_values().astype(np.float, copy=True)
np.random.shuffle(points)


## Representacion de los datos
data.plot.scatter('x', 'y', c='g')
plt.plot(group1[:,0], group1[:,1], 'rx')
plt.plot(group2[:,0], group2[:,1], 'bx')
plt.show()


## Lanza Kmeans
k = 2
iters = 5
centroids, assignments = TFKMeans(points, k, iters)

## Representa el resultado
colorByK = ['r', 'g']
for i in range(k):
	i_pts = points[filter(lambda x: assignments[x]==i, range(len(points)))]
	cent = centroids[i]
	plt.plot(cent[0], cent[1], colorByK[i]+'o')
	plt.plot(i_pts[:,0], i_pts[:,1], colorByK[i]+'.')
plt.show()

