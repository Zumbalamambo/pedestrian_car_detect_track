import numpy as np
import tensorflow as tf
import cv2
import keras
from keras.layers.core import Flatten, Dense, Activation, Reshape
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras import backend as K

# Using Tiny Yolo Net Architecture for pedestrian and car detection
class TinyYoloNet:
    # Using TensorFlow Backend for dimensions
    def __init__(self, sess):
        # Set the graph
        self.graph = sess.graph
        self.sess = sess
        K.set_session(self.sess)

        with self.graph.as_default():
            # Construct the model
            self.net = Sequential()

            # Convolutional Layer 1, 16 Feature Maps, (3,3) Kernel
            self.net.add(Conv2D(16,(3,3), input_shape=(448,448,3), padding='same', strides=(1,1)))
            self.net.add(LeakyReLU(alpha=0.1))
            self.net.add(MaxPooling2D(pool_size=(2,2)))

            # Convolutional Layer 2, 32 Feature Maps, (3,3) Kernel
            self.net.add(Conv2D(32,(3,3), padding='same'))
            self.net.add(LeakyReLU(alpha=0.1))
            self.net.add(MaxPooling2D(pool_size=(2,2), padding='valid'))

            # Convolutional Layer 3, 64 Feature Maps, (3,3) Kernel
            self.net.add(Conv2D(64,(3,3), padding='same'))
            self.net.add(LeakyReLU(alpha=0.1))
            self.net.add(MaxPooling2D(pool_size=(2,2), padding='valid'))

            # Convolutional Layer 4, 128 Feature Maps, (3,3) Kernel
            self.net.add(Conv2D(128, (3,3), padding='same'))
            self.net.add(LeakyReLU(alpha=0.1))
            self.net.add(MaxPooling2D(pool_size=(2,2), padding='valid'))

            # Convolutional Layer 5, 256 Feature Maps, (3,3) Kernel
            self.net.add(Conv2D(256,(3,3), padding='same'))
            self.net.add(LeakyReLU(alpha=0.1))
            self.net.add(MaxPooling2D(pool_size=(2,2), padding='valid'))

            # Convolutional Layer 6, 512 Feature Maps, (3,3) Kernel
            self.net.add(Conv2D(512,(3,3), padding='same'))
            self.net.add(LeakyReLU(alpha=0.1))
            self.net.add(MaxPooling2D(pool_size=(2,2), padding='valid'))

            # Convolutional Layer 7, 1024 Feature Maps, (3,3) Kernel
            self.net.add(Conv2D(1024,(3,3), padding='same'))
            self.net.add(LeakyReLU(alpha=0.1))

            # Convolutional Layer 8, 1024 Feature Maps, (3,3) Kernel
            self.net.add(Conv2D(1024,(3,3), padding='same'))
            self.net.add(LeakyReLU(alpha=0.1))

            # Convolutional Layer 9, 1024 Feature Maps, (3,3) Kernel
            self.net.add(Conv2D(1024,(3,3), padding='same'))
            self.net.add(LeakyReLU(alpha=0.1))

            # FC Layers
            self.net.add(Flatten())

            # FC Layer 10, 256 Nodes
            self.net.add(Dense(256))

            # FC Layer 11, 4096 Nodes
            self.net.add(Dense(4096))
            self.net.add(LeakyReLU(alpha=0.1))

            # FC Layer 12, 1470 Nodes
            self.net.add(Dense(1470))
        return

    # Get the weights from https://pjreddie.com/darknet/yolo/
    def set_weights(self, weight_file):
        print('Loading weights from %s...' % weight_file)
        np_data = np.fromfile(weight_file, np.float32)
        np_data = np_data[4:]

        # Just in case
        K.set_session(self.sess)
        with self.graph.as_default():
            # Iterate through the layers and assign each convolutional layer the weights and biases
            i = 0
            for layer in self.net.layers:
                shape_weights = [ w.shape for w in layer.get_weights() ]

                # Either an FC Layer or a Conv2D Layer
                if shape_weights != []:
                    weight_shape, bias_shape = shape_weights

                    # Reshape the weights into the biases
                    # Flatten the shapes with np prod to index the weights
                    bias = np_data[i:i+np.prod(bias_shape)].reshape(bias_shape)
                    i += np.prod(bias_shape)
                    weights = np_data[i:i+np.prod(weight_shape)].reshape(weight_shape)
                    i += np.prod(weight_shape)
                    layer.set_weights([weights, bias])
        return

if __name__ == "__main__":
    sess = tf.Session()
    K.set_session(sess)
    yolo = TinyYoloNet(sess)
    yolo.net.summary()
    yolo.set_weights('./weights/yolo-tiny.weights')
