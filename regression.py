import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


### LINEAR REGRESSION

## Make 100 phony data points in NumPy
x_data = np.float32(np.random.rand(2, 100))
y_data = np.dot([0.100, 0.200], x_data) + 0.300

## Read a data file
# points = np.loadtxt('points.csv')
# x_data = np.array([points[:,0], points[:,1]], dtype='float32')
# y_data = np.dot([0.100, 0.200], x_data) + 0.300

## Construct a linear model
b = tf.Variable(tf.zeros([1]))
W = tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0))
y = tf.matmul(W, x_data) + b

## Minimize the squared errors
loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

## For initializing the variables
init = tf.initialize_all_variables()

## Launch the graph
sess = tf.Session()
sess.run(init)

## Fit the plane
for step in xrange(0, 201):
    sess.run(train)
    if step % 20 == 0:
        print step, sess.run(W), sess.run(b)

## Learns best fit is W: [[0.100  0.200]], b: [0.300]


## TODO finish visualization
theta = sess.run(W)
bb = sess.run(b)
# f = lambda a: a * theta + b
# t = np.linspace(0, x_data.max() + 20, 300)
yy = sess.run(tf.matmul(W, x_data) + b)[0]
plt.plot(x_data[0], x_data[1], 'r.')
plt.plot(x_data[0], yy, 'bx')
plt.show()
