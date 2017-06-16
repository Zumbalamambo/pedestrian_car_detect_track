import numpy as np
import tiny_yolo_net
import tensorflow as tf
import cv2
import keras
import matplotlib.pyplot as plt

# Define Flags
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('video', None, 'Video file to run through the network.')
flags.DEFINE_string('image', None, 'Image file to run through the network.')
flags.DEFINE_string('record', False, 'Output of recorded yolo.')

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
    image = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, shape)
    image = np.transpose(image, (2,0,1))
    image = 2*(image/255.0) - 1
    image = np.expand_dims(image, axis=0)
    return image

def draw_box(boxes,im):
    for b in boxes:
        h, w, _ = im.shape
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

        cv2.rectangle(im, (left, top), (right, bot), (255,0,0), thick)

    return im

# Run the network here
if __name__ == "__main__":
    # Set the network
    print('Instantiating Tiny Yolo Net...')
    yolo = tiny_yolo_net.TinyYoloNet()
    #yolo.set_weights('./weights/yolo-tiny.weights')
    load_weights(yolo.net, './weights/yolo-tiny.weights')

    if FLAGS.video != None:
        print('Processing video...')
        cap = cv2.VideoCapture(FLAGS.video)

        # Set Recording parameters
        if FLAGS.record != None:
            fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
            out = cv2.VideoWriter(FLAGS.record, fourcc, 29.0, (488,488))

        while(cap.isOpened()):
            ret, frame = cap.read()
            # Break if no more
            if ret == False:
                break

            # Yolo Processing
            proc_frame = proc_load_image(frame, (448,448))
            boxes = yolo.process(proc_frame, thresholds=[0.17,0.17], classes=[6,14], iou_threshold=0.4)
            boxed_frame = draw_box(boxes,cv2.resize(frame,(488,488)))

            # Save to file
            if FLAGS.record != None:
                out.write(boxed_frame)

            # Display
            #cv2.imshow('Yolo Out', boxed_frame)
            """
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            """
        print('Finished!')
        cap.release()
        cv2.destroyAllWindows()

    elif FLAGS.image != None:
        print('Processing video...')
        # Test
        proc_im = proc_load_image(cv2.imread(FLAGS.image), (448,448))
        im = cv2.resize(cv2.imread(FLAGS.image), (448,448))
        boxes = yolo.process(proc_im, thresholds=[0.1,0.1], classes=[6,14], iou_threshold=0.4)
        boxed_im = draw_box(boxes,im)
        plt.imsave('out.jpg', boxed_im)
    else:
        print('No file specified')
