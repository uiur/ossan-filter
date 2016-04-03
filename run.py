import sys

import tensorflow as tf
from deep import *
import data

if len(sys.argv) < 2:
    print("python run.py /path/to/image.png")

def read_image(path):
    image_data = tf.read_file(path)
    image = tf.image.decode_png(image_data, channels=3)
    image.set_shape(data.IMAGE_SHAPE)
    resized = tf.image.resize_image_with_crop_or_pad(image, data.SIZE, data.SIZE)
    return data.normalize(tf.cast(resized, tf.float32))

image = read_image(sys.argv[1])
predictions = tf.nn.softmax(inference(tf.expand_dims(image, 0)))

sess = tf.Session()
tf.train.start_queue_runners(sess=sess)

saver = tf.train.Saver()
saver.restore(sess, 'tmp/train/-9999')

prob = sess.run(predictions)
print("ossan: %f\nother: %f" % (prob[0][0], prob[0][1]))
