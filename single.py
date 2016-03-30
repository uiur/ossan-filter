import tensorflow as tf
import numpy as np

sess = tf.Session()

# 24 x 24
x = tf.placeholder(tf.float32, shape=[None, 24 * 24])
y_ = tf.placeholder(tf.float32, shape=[None, 2])

W = tf.Variable(tf.zeros([24 * 24, 2]))
b = tf.Variable(tf.zeros([2]))

y = tf.nn.softmax(tf.matmul(x,W) + b)
cross_entropy = -tf.reduce_sum(y_*tf.log(y))

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
sess.run(tf.initialize_all_variables())

for i in range(10000):
  batch = next_batch(50)
  sess.run(train_step, feed_dict={x: batch[0], y_: batch[1]})

  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  test_images, test_labels = test
  if i % 100 == 0:
      a = sess.run(accuracy, feed_dict={x: test_images, y_: test_labels})
      print('Epoch: %d' % i)
      print('Accuracy: %0.04f' % a)
