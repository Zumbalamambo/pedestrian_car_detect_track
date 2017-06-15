import numpy as np
import tiny_yolo_net
import tensorflow as tf
import cv2
from keras import backend as K

# Will load and image and pre-process it for yolo_net
def load_image(im, shape):
    image = cv2.imread(im)
    image = cv2.resize(image, shape)
    image = np.expand_dims(image, axis=0)
    return image

if __name__ == "__main__":
    # Make the Session
    sess = tf.Session()
    K.set_session(sess)

    # Set the network
    print('Instantiating Tiny Yolo Net...')
    yolo = tiny_yolo_net.TinyYoloNet(sess)
    yolo.net.summary()
    yolo.set_weights('./weights/yolo-tiny.weights')

    # Test
    test_im = load_image('./data/1.jpg', (448,448))
    out = yolo.net.predict(test_im)
    print(out)
