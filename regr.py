import tensorflow as tf
import numpy as np


n = 10
c1 = 0.7
c0 = 9.0
iters = 100


## Datos originales
x_data = np.linspace(0, 10, n, dtype='float32')
y_data = c1 * x_data + c0


## Definicion del modelo
x = tf.constant(x_data)
w1 = tf.Variable(tf.zeros([1], 'float32'), name='w1')
w0 = tf.Variable(tf.zeros([1], 'float32'), name='w0')
y = tf.add(tf.mul(x, w1), w0)


## Algoritmo de entrenamiento
cost = tf.reduce_mean(tf.square(y - y_data))
train = tf.train.GradientDescentOptimizer(learning_rate=0.015).minimize(cost)


## Ejecucion del aprendizaje
init = tf.initialize_all_variables()
sess = tf.InteractiveSession()
sess.run(init)
for i in range(iters):
	sess.run(train)
	result = sess.run([w1, w0, cost])
	print "w1:", result[0], ", w0:", result[1], ", con coste", result[2]


## Datos predecidos
y_learn = y.eval()


## Guarda el grafo para TensorBoard
writer = tf.python.training.summary_io.SummaryWriter('/Users/sergio/Desktop/tensorflow/')
writer.add_graph(sess.graph_def)
writer.close()


sess.close()

