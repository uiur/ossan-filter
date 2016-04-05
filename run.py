import sys

import tensorflow as tf
import cv2
from deep import *
import data

cascade_file = "/usr/local/Cellar/opencv/2.4.12_2/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml"

def read_face(path):
    image = cv2.imread(path)
    cascade = cv2.CascadeClassifier(cascade_file)
    faces = cascade.detectMultiScale(image,
                                     scaleFactor = 1.1,
                                     minNeighbors = 5,
                                     minSize = (28, 28))
    (x, y, w, h) = faces[0]
    return image[y:y+h, x:x+w]

image_input = tf.placeholder(tf.float32, shape=data.IMAGE_SHAPE)

class Model:
    def __init__(self):
        self.predictions = tf.nn.softmax(inference(tf.expand_dims(image_input, 0)))
        self.sess = tf.Session()

        saver = tf.train.Saver()
        saver.restore(self.sess, 'tmp/train/0403-9999')

    def resize(self, image):
        resized = tf.image.resize_image_with_crop_or_pad(tf.constant(image), data.SIZE, data.SIZE)
        return data.normalize(tf.cast(resized, tf.float32))

    def predict(self, raw_image):
        resized = self.resize(raw_image)
        image = self.sess.run(resized)
        return self.sess.run(self.predictions, feed_dict={image_input: image})

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("python run.py /path/to/image.png")

    face = read_face(sys.argv[1])
    model = Model()
    prob = model.predict(face)

    print("ossan: %f\nother: %f" % (prob[0][0], prob[0][1]))
