import glob
import tensorflow as tf

SIZE = 28
IMAGE_SHAPE = [SIZE, SIZE, 3]

def normalize(image):
    return image / 255.0 - 0.5

def load_image(pattern):
    filenames = glob.glob(pattern)
    queue = tf.train.string_input_producer(filenames)

    reader = tf.WholeFileReader()
    key, value = reader.read(queue)
    image = tf.image.decode_png(value, channels=3)
    return normalize(tf.cast(image, tf.float32))

def load_ossan():
    image = load_image('resized_data/train/ossan/*.png')
    return tf.tuple([image, tf.constant([1., 0.])])

def load_other():
    image = load_image('resized_data/train/other/*.png')
    return tf.tuple([image, tf.constant([0., 1.])])

def load_test(batch_size=300):
    ossan = load_image('resized_data/test/ossan/*.png')
    other = load_image('resized_data/test/other/*.png')

    return tf.train.batch_join([tf.tuple([ossan, tf.constant([1., 0.])]), tf.tuple([other, tf.constant([0., 1.])])], batch_size, shapes=[IMAGE_SHAPE, [2]])

def batch(batch_size):
    return tf.train.shuffle_batch_join([load_ossan(), load_other()], batch_size, 1000, 10, enqueue_many=False, shapes=[IMAGE_SHAPE, [2]])
