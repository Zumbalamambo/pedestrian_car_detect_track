import numpy as np
import tiny_yolo_net
import tensorflow as tf
import cv2
import keras
import matplotlib.pyplot as plt

# Define Flags
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('video', None, 'Video file to run through the network')
flags.DEFINE_string('image', None, 'Image file to run through the network')

def load_weights(model,yolo_weight_file):
    data = np.fromfile(yolo_weight_file,np.float32)
    data=data[4:]

    index = 0
    for layer in model.layers:
        shape = [w.shape for w in layer.get_weights()]
        if shape != []:
            kshape,bshape = shape
            bia = data[index:index+np.prod(bshape)].reshape(bshape)
            index += np.prod(bshape)
            ker = data[index:index+np.prod(kshape)].reshape(kshape)
            index += np.prod(kshape)
            layer.set_weights([ker,bia])


# Will load and image and pre-process it for yolo_net
def proc_load_image(im, shape):
    image = cv2.cvtColor(cv2.imread(im), cv2.COLOR_BGR2RGB)
    image = cv2.imread(im)
    image = cv2.resize(image, shape)
    image = np.transpose(image, (2,0,1))
    image = 2*(image/255.0) - 1
    image = np.expand_dims(image, axis=0)
    return image

def draw_box(boxes,im):
    imgcv = im

    for b in boxes:
        h, w, _ = imgcv.shape
        bx = b[0]; by = b[1]; bw = b[2]; bh = b[3];

        left  = int ((bx - bw/2.) * w)
        right = int ((bx + bw/2.) * w)
        top   = int ((by - bh/2.) * h)
        bot   = int ((by + bh/2.) * h)
        if left  < 0    :  left = 0
        if right > w - 1: right = w - 1
        if top   < 0    :   top = 0
        if bot   > h - 1:   bot = h - 1
        thick = int((h + w) // 150)

        cv2.rectangle(imgcv, (left, top), (right, bot), (255,0,0), thick)

    return imgcv

# Run the network here
if __name__ == "__main__":
    # Make the Session

    # Set the network
    keras.backend.set_image_dim_ordering('th')
    print('Instantiating Tiny Yolo Net...')
    yolo = tiny_yolo_net.TinyYoloNet()
    #yolo.set_weights('./weights/yolo-tiny.weights')
    load_weights(yolo.net, './weights/yolo-tiny.weights')

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
    #proc_im = proc_load_image('./data/test1.jpg', (448,448))
    imagePath = './data/1.jpg'
    image = plt.imread(imagePath)

    resized = cv2.resize(image,(448,448))
    batch = np.transpose(resized,(2,0,1))
    batch = 2*(batch/255.) - 1
    batch = np.expand_dims(batch, axis=0)
    proc_im = batch

    im = cv2.imread('./data/test1.jpg')
    boxes = yolo.process(proc_im)

    print(boxes)

    test_im = draw_box(boxes,resized)
    plt.imsave('out.jpg', test_im)
