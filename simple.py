
import tensorflow as tf


## Constants
# NUM_CORES = 2		# Number of cores to use
NUM_X = 111111111111111111
NUM_Y = 555555555555555555


## Declares the inputs for the operation
x = tf.placeholder('int64')
y = tf.placeholder('int64')


## Declares the operation
xy = tf.math_ops.add(x, y)


## Creates a session and initiliza the variables
sess = tf.Session()
# sess = tf.Session(tf.ConfigProto(inter_op_parallelism_threads=NUM_CORES,
# 		intra_op_parallelism_threads=NUM_CORES))
sess.run(tf.initialize_all_variables())


## Makes the sum
print sess.run(xy, feed_dict={x: NUM_X, y: NUM_Y})



# writer = tf.python.training.summary_io.SummaryWriter('/Users/sergio/Desktop/TensorFlow/')
# writer.add_graph(sess.graph_def)
