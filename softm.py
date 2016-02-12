import tensorflow as tf
import input_data


## Carga de los datos
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

## Definicion de parametros
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

## Definicion del modelo
y = tf.nn.softmax(tf.matmul(x, W) + b)
y_ = tf.placeholder(tf.float32, [None, 10])	# valores originales

## Funcion de coste y entrenamiento
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
train = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

## Inicializa la ejecucion
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

## Lanza el entrenamiento
for i in range(500):
	batch_xs, batch_ys = mnist.train.next_batch(100)
	sess.run(train, feed_dict={x: batch_xs, y_: batch_ys})


## Evalua el modelo aprendido
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print "Accuracy", sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})

writer = tf.python.training.summary_io.SummaryWriter('/Users/sergio/Desktop/tensorflow/')
writer.add_graph(sess.graph_def)
writer.close()
sess.close()

