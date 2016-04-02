import glob
import cv2
import numpy as np
import random

SIZE = 28

# grayscale
def read_images(files):
    return [cv2.resize(cv2.imread(f, 0), (SIZE, SIZE)) for f in files]

def make_data(ossan_images, other_images):
    input_data = np.array([image.flatten() for image in ossan_images + other_images], dtype='float32')
    input_data = input_data / 255.0 - 0.5
    label_data = np.concatenate(
        (
            np.array([(1., 0.) for i in range(len(ossan_images))]),
            np.array([(0., 1.) for i in range(len(other_images))]),
        )
    )

    return (input_data, label_data)

def load_data():
    ossan_images = read_images(glob.glob('./data/ossan/*.png'))
    np.random.shuffle(ossan_images)
    test_ossan_images, training_ossan_images = ossan_images[0:100], ossan_images[101:]

    other_images = read_images(glob.glob('./data/other/*.png'))
    np.random.shuffle(other_images)
    test_other_images, training_other_images = other_images[0:100], other_images[101:]

    training = make_data(training_ossan_images, training_other_images)
    test = make_data(test_ossan_images, test_other_images)
    return (training, test)

def next_batch(data, size):
    inputs = []
    labels = []
    for i in range(size):
        random_index = random.randint(0, len(data[0]) - 1)
        inputs.append(data[0][random_index])
        labels.append(data[1][random_index])

    return (np.array(inputs), np.array(labels))
