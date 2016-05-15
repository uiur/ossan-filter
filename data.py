import glob
import tensorflow as tf

SIZE = 28
IMAGE_SHAPE = [SIZE, SIZE, 3]

def normalize(image):
    return image / 255.0 - 0.5

def load_image(pattern, distort=True):
    filenames = glob.glob(pattern)
    queue = tf.train.string_input_producer(filenames)

    reader = tf.WholeFileReader()
    key, value = reader.read(queue)
    image = tf.image.decode_png(value, channels=3)
    image.set_shape(IMAGE_SHAPE)
    image = tf.cast(image, tf.float32)
    if distort:
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, max_delta=0.4)
        image = tf.image.random_contrast(image, lower=0.6, upper=1.4)
        image = tf.image.random_hue(image, max_delta=0.04)
        image = tf.image.random_saturation(image, lower=0.6, upper=1.4)

    return normalize(image)

def load_ossan():
    image = load_image('data/train/ossan/*.png')
    return tf.tuple([image, tf.constant([1., 0.])])

def load_other():
    image = load_image('data/train/other/*.png')
    return tf.tuple([image, tf.constant([0., 1.])])

def load_test(batch_size=450):
    ossan = load_image('data/test/ossan/*.png', distort=False)
    other = load_image('data/test/other/*.png', distort=False)

    return tf.train.batch_join([
        tf.tuple([ossan, tf.constant([1., 0.])]),
        tf.tuple([other, tf.constant([0., 1.])])
    ], batch_size, shapes=[IMAGE_SHAPE, [2]])

def batch(batch_size):
    return tf.train.shuffle_batch_join([load_ossan(), load_other()], batch_size, 1000, 10, enqueue_many=False, shapes=[IMAGE_SHAPE, [2]])
