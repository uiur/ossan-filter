import tensorflow as tf
import numpy as np
import data
import math

input_size = data.SIZE * data.SIZE

def show_image(img):
    img = (img + 0.5) * 255
    cv2.imshow('image', cv2.cvtColor(img.reshape([data.SIZE, data.SIZE]).astype('uint8'), cv2.COLOR_GRAY2BGR))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def init_weight(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=1.0 / math.sqrt(input_size)))

def init_bias(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=1.0))

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def inference(images):
    W_conv1 = init_weight([5, 5, 1, 32])
    b_conv1 = init_bias([32])

    x_image = tf.reshape(x, [-1, data.SIZE, data.SIZE, 1])

    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    W_conv2 = init_weight([5, 5, 32, 64])
    b_conv2 = init_bias([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    W_fc1 = init_weight([7 * 7 * 64, 1024])
    b_fc1 = init_bias([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = init_weight([1024, 2])
    b_fc2 = init_bias([2])

    return tf.matmul(h_fc1_drop, W_fc2) + b_fc2

def loss(logits, labels):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, labels))

def train(total_loss):
    ce_summ = tf.scalar_summary("loss", total_loss)

    return tf.train.AdamOptimizer().minimize(total_loss)

def do_eval(logits, labels):
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    accuracy_summary = tf.scalar_summary("accuracy", accuracy)

    return accuracy

training, test = data.load_data()

x = tf.placeholder(tf.float32, shape=[None, input_size])
y_ = tf.placeholder(tf.float32, shape=[None, 2])
keep_prob = tf.placeholder(tf.float32)

predictions = inference(x)
total_loss = loss(predictions, y_)
train_step = train(total_loss)

saver = tf.train.Saver()
sess = tf.Session()

sess.run(tf.initialize_all_variables())

merged = tf.merge_all_summaries()
writer = tf.train.SummaryWriter("./tmp/logs", sess.graph_def)

for i in range(10000):
  batch = data.next_batch(training, 50)
  sess.run(train_step, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

  test_images, test_labels = test
  if i % 100 == 0:
      accuracy = do_eval(predictions, y_)
      train_accuracy = sess.run(accuracy, feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0 })
      (m, test_accuracy) = sess.run([merged, accuracy], feed_dict={x: test_images, y_: test_labels, keep_prob: 1.0 })

      writer.add_summary(m, i)

      print('Epoch %d:  train: %0.04f   test: %0.04f' % (i, train_accuracy, test_accuracy))

  if (i+1) % 1000 == 0:
     saver.save(sess, './tmp/train/', global_step=i)

test_images, test_labels = test
print(sess.run(accuracy, feed_dict={x: test_images, y_: test_labels, keep_prob: 1.0}))
