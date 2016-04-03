import sys

import tensorflow as tf
import cv2
from deep import *
import data

if len(sys.argv) < 2:
    print("python run.py /path/to/image.png")

cascade_file = "/usr/local/Cellar/opencv/2.4.12_2/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml"

def read_face(path):
    image = cv2.imread(path)
    cascade = cv2.CascadeClassifier(cascade_file)
    faces = cascade.detectMultiScale(image,
                                     scaleFactor = 1.1,
                                     minNeighbors = 5,
                                     minSize = (28, 28))
    return faces[0]

def read_image(path):
    image = tf.constant(read_face(path), shape=data.IMAGE_SHAPE)

    resized = tf.image.resize_image_with_crop_or_pad(image, data.SIZE, data.SIZE)
    return data.normalize(tf.cast(resized, tf.float32))

image = read_image(sys.argv[1])
predictions = tf.nn.softmax(inference(tf.expand_dims(image, 0)))

sess = tf.Session()
tf.train.start_queue_runners(sess=sess)

saver = tf.train.Saver()
saver.restore(sess, 'tmp/train/0403-9999')

prob = sess.run(predictions)
print("ossan: %f\nother: %f" % (prob[0][0], prob[0][1]))
