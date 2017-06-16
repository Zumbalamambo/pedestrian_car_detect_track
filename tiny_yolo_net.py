#############################################################################################################
# Referenced this to implement yolonet https://github.com/xslittlegrass/CarND-Vehicle-Detection
#############################################################################################################
import numpy as np
import tensorflow as tf
import cv2
import keras
import matplotlib.pyplot as plt
from functools import reduce
from keras.layers.core import Flatten, Dense, Activation, Reshape
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.models import Sequential
from keras import backend as K

# Calculate intersection of two bounding boxes
def intersection(x1, y1, w1, h1,x2, y2, w2, h2):
    # Calculate width of intersecting region
    l1 = x1 - w1 / 2.0; l2 = x2 - w2 / 2.0;
    l = max(l1,l2)
    r1 = x1 + w1 / 2.0; r2 = x2 + w2 / 2.0;
    r = min(r1, r2)
    w = r - l
    # Calculate height of intersecting region
    b1 = y1 - h1 / 2.0; b2 = y2 - h2 / 2.0;
    b = max(b1, b2)
    t1 = y1 + h1 / 2.0; t2 = y2 + h2 / 2.0;
    t = min(t1, t2)
    h = t - b
    return w*h if (w >= 0 and h >= 0) else 0

# Calculate union of two bounding boxes
def union(x1, y1, w1, h1, x2, y2, w2, h2):
    return (w1*h1 + w2*h2 - intersection(x1,y1,w1,h1,x2,y2,w2,h2))

# Perform intersection over union in order to evaluate our bounding boxes
def iou(x1, y1, w1, h1, x2, y2, w2, h2):
    return (intersection(x1,y1,w1,h1,x2,y2,w2,h2) / union(x1,y1,w1,h1,x2,y2,w2,h2))

# Using Tiny Yolo Net Architecture for pedestrian and car detection
class TinyYoloNet:
    def __init__(self):
        # Output Parameters
        self.grid_len = 7           # Length of X and Y axis for grid boxes
        self.num_classes = 20       # Number of classes from the trained VOC
        self.num_boxes = 2   # Confidences per box

        self.num_grid_cells = self.grid_len * self.grid_len                     # Should be 49
        self.probability_len = self.num_grid_cells * self.num_classes           # Should be 980, 20 class probabilities per grid cell
        self.confidence_len = self.num_grid_cells * self.num_boxes              # Should be 98, 2 Confidences per grid cell
        self.boxes_len = 1470 - self.probability_len - self.confidence_len      # Remaining are box coordinates

        # Set the graph
        # Construct the model
        keras.backend.set_image_dim_ordering('th')
        self.net = Sequential()

        # Convolutional Layer 1, 16 Feature Maps, (3,3) Kernel
        self.net.add(Convolution2D(16,3,3, input_shape=(3,448,448), border_mode='same', subsample=(1,1)))
        self.net.add(LeakyReLU(alpha=0.1))
        self.net.add(MaxPooling2D(pool_size=(2,2)))

        # Convolutional Layer 2, 32 Feature Maps, (3,3) Kernel
        self.net.add(Convolution2D(32,3,3, border_mode='same'))
        self.net.add(LeakyReLU(alpha=0.1))
        self.net.add(MaxPooling2D(pool_size=(2,2), border_mode='valid'))

        # Convolutional Layer 3, 64 Feature Maps, (3,3) Kernel
        self.net.add(Convolution2D(64,3,3, border_mode='same'))
        self.net.add(LeakyReLU(alpha=0.1))
        self.net.add(MaxPooling2D(pool_size=(2,2), border_mode='valid'))

        # Convolutional Layer 4, 128 Feature Maps, (3,3) Kernel
        self.net.add(Convolution2D(128, 3,3, border_mode='same'))
        self.net.add(LeakyReLU(alpha=0.1))
        self.net.add(MaxPooling2D(pool_size=(2,2), border_mode='valid'))

        # Convolutional Layer 5, 256 Feature Maps, (3,3) Kernel
        self.net.add(Convolution2D(256,3,3, border_mode='same'))
        self.net.add(LeakyReLU(alpha=0.1))
        self.net.add(MaxPooling2D(pool_size=(2,2), border_mode='valid'))

        # Convolutional Layer 6, 512 Feature Maps, (3,3) Kernel
        self.net.add(Convolution2D(512,3,3, border_mode='same'))
        self.net.add(LeakyReLU(alpha=0.1))
        self.net.add(MaxPooling2D(pool_size=(2,2), border_mode='valid'))

        # Convolutional Layer 7, 1024 Feature Maps, (3,3) Kernel
        self.net.add(Convolution2D(1024,3,3, border_mode='same'))
        self.net.add(LeakyReLU(alpha=0.1))

        # Convolutional Layer 8, 1024 Feature Maps, (3,3) Kernel
        self.net.add(Convolution2D(1024,3,3, border_mode='same'))
        self.net.add(LeakyReLU(alpha=0.1))

        # Convolutional Layer 9, 1024 Feature Maps, (3,3) Kernel
        self.net.add(Convolution2D(1024,3,3, border_mode='same'))
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
        # Output Parameters
        print('Loading weights from %s...' % weight_file)
        np_data = np.fromfile(weight_file, np.float32)
        np_data = np_data[4:]

        # Just in case
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

    # Process an image
    # Output: [[box_info_0], ..., [box_info_n]]
    # box_info_i = [x,y,w,h,c,prob, label]
    def process(self, image, threshold_ious=[0.1,0.1], classes=[6,14]):
        yolo_out = self.net.predict(image)[0]

        # Split the output vector into the corresponding data formats
        probabilities = yolo_out[0:self.probability_len].reshape((self.num_grid_cells, self.num_classes))
        confidences = yolo_out[self.probability_len:self.probability_len+self.confidence_len].reshape((self.num_grid_cells, self.num_boxes))
        box_coordinates = yolo_out[self.probability_len + self.confidence_len:].reshape((self.num_grid_cells, self.num_boxes, 4)) # 4 because 4 values per box

        # Get the boxes
        boxes = []
        for grid_cell in range(self.num_grid_cells):
            for i in range(self.num_boxes):
                # Grab all confidence value and probability,
                p = probabilities[grid_cell, :] * confidences[grid_cell,i]

                # Filter out box that doesn't meet threshold
                threshold_res = [ (label, p[label]) for label, threshold in zip(classes, threshold_ious) if p[label] >= threshold]

                # Sort by probability value in tuple
                threshold_res.sort(key=lambda x: x[1], reverse=True)

                if threshold_res == []:
                    continue

                # Get the box
                boxes.append([
                    # Grab the coordinate values for the boxes
                    (box_coordinates[grid_cell, i, 0] + grid_cell % self.grid_len) / self.grid_len,
                    (box_coordinates[grid_cell, i, 1] + grid_cell // self.grid_len) / self.grid_len,
                    (box_coordinates[grid_cell, i, 2]) ** 1.8,
                    (box_coordinates[grid_cell, i, 3]) ** 1.8,
                    confidences[grid_cell,i],
                    threshold_res
                ])

        # Merge boxes now
        # Must merge with multi-class labels

        return boxes

# Run the network here
if __name__ == "__main__":
    yolo = TinyYoloNet()
    yolo.net.summary()
    yolo.set_weights('./weights/yolo-tiny.weights')

    # Test set operations
    x1 = 0; y1 = 0; w1 = 4; h1 = 4;
    x2 = 2; y2 = 2; w2 = 4; h2 = 4;

    print ('Test union of (0,0,4,4), (2,2,4,4), Should be 28, results: %d' % union(x1,y1,w1,h1,x2,y2,w2,h2))
