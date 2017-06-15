import numpy as np
import tiny_yolo_net
import tensorflow as tf
import cv2
from keras import backend as K

if __name__ == "__main__":
    # Make the Session
    sess = tf.Session()
    K.set_session(sess)

    # Set the network
    print('Instantiating Tiny Yolo Net...')
    yolo = tiny_yolo_net.TinyYoloNet(sess)
    yolo.net.summary()
    yolo.set_weights('./weights/yolo-tiny.weights')

    # Run the network
    cv2.
    yolo.net.predict()
