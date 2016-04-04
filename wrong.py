import tensorflow as tf
from deep import *
import data

import numpy as np
import PIL.Image
from IPython.display import Image, display

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('model', 'tmp/train/0403-9999', '')

sess = tf.Session()

test_images, test_labels = data.load_test()
predictions = tf.nn.softmax(inference(test_images))

wrong_prediction = tf.not_equal(tf.argmax(predictions, 1), tf.argmax(test_labels, 1))
wrong_images = tf.boolean_mask(test_images, wrong_prediction)

saver = tf.train.Saver()
saver.restore(sess, FLAGS.model)
tf.train.start_queue_runners(sess=sess)

images = sess.run((wrong_images + 0.5) * 255.0)
images = np.uint8(np.clip(images, 0, 255))

i = 0
for image in images:
    PIL.Image.fromarray(image).save("tmp/wrong-%d.png" % i)
    i += 1
