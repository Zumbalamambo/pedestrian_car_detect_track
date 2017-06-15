import numpy as np
import tiny_yolo_net
import tensorflow as tf
import cv2
from keras import backend as K

# Define Flags
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('video', None, 'Video file to run through the network')
flags.DEFINE_string('image', None, 'Image file to run through the network')

# Will load and image and pre-process it for yolo_net
def proc_load_image(im, shape):
    image = cv2.imread(im)
    image = cv2.resize(image, shape)
    image = np.expand_dims(image, axis=0)
    return image

# Run the network here
if __name__ == "__main__":
    # Make the Session
    sess = tf.Session()
    K.set_session(sess)

    # Set the network
    print('Instantiating Tiny Yolo Net...')
    yolo = tiny_yolo_net.TinyYoloNet(sess)
    yolo.net.summary()
    yolo.set_weights('./weights/yolo-tiny.weights')

    """
    if FLAGS.video != None:
        print('Processing video...')
    elif FLAGS.image != None:
        print('Processing video...')
        # Test
        test_im = proc_load_image('./data/1.jpg', (448,448))
    else:
        print('No file specified')
    """

    # Test
    test_im = proc_load_image('./data/1.jpg', (448,448))

    boxes = yolo.process(test_im)
