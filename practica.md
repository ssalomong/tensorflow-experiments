---
title: Práctica con Tensorflow
author: Sergio Salomón García
---


# Introducción

API oficial de TensorFlow:
> https://www.tensorflow.org/versions/v0.6.0/api_docs/index.html


En primer lugar, importaremos todos los módulos que vamos a necesitar:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
```


-------

Cálculo de multiplicación de matrices:
```python
matrix1 = tf.constant([[8., 8.]])
matrix2 = tf.constant([[2.],[2.]])
product = tf.matmul(matrix1, matrix2)
```


Diversas formas de ejecutar un grafo:
```python
# Creacion de una sesion (al terminar debe cerrarse y con variables se complica)
sess = tf.Session()
print sess.run(product)
sess.close()

# Con el bloque with (se cierra automaticamente)
with tf.Session() as sess:
	print sess.run(product)

# Con una sesion interactiva (permite evaluar un nodo directamente)
sess = tf.InteractiveSession()
print product.eval()
sess.close()
```

-------



Una prueba sencilla para sumar dos *constantes*:
```python
a = tf.constant(11)
b = tf.constant(31)
with tf.Session() as sess:
	print sess.run(a + b)
```


Si en el caso anterior se utilizasen *placeholders*:
```python
a = tf.placeholder('int32')
b = tf.placeholder('int32')
with tf.Session() as sess:
	print sess.run(a + b, feed_dict={a: 11, b: 31})
```


Ahora utilizamos *variables* para la misma suma:
```python
a = tf.Variable(tf.zeros([1], 'int32'))
b = tf.Variable(tf.zeros([1], 'int32'))
with tf.Session() as sess:
	sess.run(tf.initialize_all_variables())
	sess.run(a.assign(tf.constant([11])))
	sess.run(b.assign(tf.constant([31])))
	print sess.run(a + b)
```

------



# Ejemplo de regresion lineal

Se define el modelo de regresión lineal a partir de un conjunto de puntos generado (x_data, y_data).
Queremos aprender los coeficientes que determinan la recta $y = w_1 x + w_0$.
```python
n = 10
x_data = np.linspace(0, 10, n, dtype='float32')

c1 = 0.7
c0 = 9.0
y_data = c1 * x_data + c0

x = tf.constant(x_data)
w1 = tf.Variable(tf.zeros([1], 'float32'), name='w1')
w0 = tf.Variable(tf.zeros([1], 'float32'), name='w0')
y = tf.add(tf.mul(x, w1), w0)	# equivalente a y = x * w1 + w0
```


Para entrenar nuestro modelo, vamos a minimizar el error cuadrático medio con el método del gradiente.
```python
cost = tf.reduce_mean(tf.square(y - y_data))
train = tf.train.GradientDescentOptimizer(learning_rate=0.015).minimize(cost)
```

Finalmente, se inicializan las variables y se realiza el entrenamiento un número de iteraciones.
```python
iters = 100

init = tf.initialize_all_variables()
sess = tf.InteractiveSession()
sess.run(init)
for i in range(iters):
	sess.run(train)
	result = sess.run([w1, w0, cost])
	print "w1:", result[0], ", w0:", result[1], ", con coste", result[2]

y_learn = y.eval()
```


Probamos a visualizar el grafo en TensorBoard, guardando la información del grafo en el directorio actual ('./'):
```python
writer = tf.python.training.summary_io.SummaryWriter('/Users/sergio/Desktop/tensorflow/')
writer.add_graph(sess.graph_def)
writer.close()
```

TensorBoard se lanza desde terminal con el comando:
```
tensorboard --logdir /Users/sergio/Desktop/tensorflow/
```



# Ejemplo de clustering

Primero construimos varios conjuntos de puntos aleatorios para poder aplicar sobre estos segmentación. Para esto, generaremos puntos de distintas distribuciones gaussianas multivariantes dadas por sus medias y matriz de covarianza.
```python
samples = 20
cov = np.array([[12,4], [4,14]])
group1 = np.random.multivariate_normal((37,33), cov, samples)
group2 = np.random.multivariate_normal((19,21), cov, samples)

data = pd.DataFrame(np.concatenate((group1, group2)), columns=['x','y'])
points = data.get_values().astype(np.float, copy=True)
np.random.shuffle(points)
```



Visualizamos los datos mediante Pandas y Matplotlib.
```python
data.plot.scatter('x', 'y', c='g')
plt.plot(group1[:,0], group1[:,1], 'rx')
plt.plot(group2[:,0], group2[:,1], 'bx')
```


Podemos analizar la siguiente adaptación del algoritmo KMeans con TensorFlow, cuya versión original fue obtenida de [aquí](https://codesachin.wordpress.com/2015/11/14/k-means-clustering-with-tensorflow/).
```python
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
		return centroids, assignments
```


Lanzamos el KMeans y visualizamos los datos para comprobar el resultado de la segmentación.
```python
k = 2
iters = 5
centroids, assignments = TFKMeans(points, k, iters)

colorByK = ['r', 'g']
for i in range(k):
	i_pts = points[filter(lambda x: assignments[x]==i, range(len(points)))]
	cent = centroids[i]
	plt.plot(cent[0], cent[1], colorByK[i]+'o')
	plt.plot(i_pts[:,0], i_pts[:,1], colorByK[i]+'.')
```


Cabe destacar que TensorFlow posee más [métodos útiles](https://www.tensorflow.org/versions/v0.6.0/api_docs/python/math_ops.html#segmentation) para la tarea de segmentación, aunque estos no sean empleados en la anterior implementación de KMeans.
```python
pts = tf.constant([1, 2, 3, 4, 5, 7], 'float32')
assignments = tf.constant([0, 0, 0, 1, 1, 1])	# k = 2
tf.segment_mean(pts, assignments).eval()
```


# Ejemplo de clasificación y Softmax

> https://www.tensorflow.org/versions/v0.6.0/tutorials/mnist/beginners/index.html

```python
import tensorflow as tf
import input_data


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, W) + b)

y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
train = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

for i in range(500):
	batch_xs, batch_ys = mnist.train.next_batch(100)
	sess.run(train, feed_dict={x: batch_xs, y_: batch_ys})


correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print "Accuracy", sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})

```
