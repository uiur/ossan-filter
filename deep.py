import tensorflow as tf
import data
import math

input_size = data.SIZE * data.SIZE

def init_weight(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=1.0 / math.sqrt(input_size)))

def init_bias(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=1.0))

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def inference(images, keep_prob=tf.constant(1.0)):
    W_conv1 = init_weight([5, 5, 3, 32])
    b_conv1 = init_bias([32])

    h_conv1 = tf.nn.relu(conv2d(images, W_conv1) + b_conv1)
    h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    W_conv2 = init_weight([5, 5, 32, 64])
    b_conv2 = init_bias([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

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

def evaluate(logits, labels):
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    accuracy_summary = tf.scalar_summary("accuracy", accuracy)

    return accuracy
