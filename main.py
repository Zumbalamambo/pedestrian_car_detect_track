import numpy as np
import tiny_yolo_net
import tensorflow as tf
import cv2
import keras
import sys

# Define Flags
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('video', None, 'Video file to run through the network.')
flags.DEFINE_string('image', None, 'Image file to run through the network.')
flags.DEFINE_string('record', None, 'Output of recorded yolo.')
flags.DEFINE_string('alg', 0, 'Tracking algorithm to run.')
flags.DEFINE_string('detect_rate', 50, 'Rerun detection after this many frames')

# define tracking algorithm options
tracker_type = ["MIL", "BOOSTING", "MEDIANFLOW", "TLD", "KCF"]

# List of tracker objects
trackers = []


# Will load and image and pre-process it for yolo_net
def proc_load_image(im, shape):
    image = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, shape)
    image = np.transpose(image, (2,0,1))
    image = 2*(image/255.0) - 1
    image = np.expand_dims(image, axis=0)
    return image


def draw_box(boxes,im):
    proc_boxes = []
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

        #print ("left: ", left, " right: ", right, " top: ", top, " bot: ", bot)
        #print
        cv2.rectangle(im, (left, top), (right, bot), (255,0,0), thick)
        proc_boxes.append((left,top,right,bot))

    return im, proc_boxes


def draw_tracker_box(boxes, img):
    for i in range(len(boxes)):
        p1 = (int(boxes[i][0]), int(boxes[i][1]))
        p2 = (int(boxes[i][0] + boxes[i][2]), int(boxes[i][1] + boxes[i][3]))
        cv2.rectangle(img, p1, p2, (0,0,255), 5)

    return img


def init_tracker(boxes, frame, alg):
    # cleanup previous tracker objects (if neccessary)
    if len(trackers) > 0:
        del trackers[:]

    # Create and initialize one tracker for each object
    for i in range(len(boxes)):
        trackers.append(cv2.Tracker_create(alg))
        ok = trackers[i].init(frame, boxes[i])

        if(ok == False):
            print("Couldn't initialize tracker for object ", i)
        else:
            print("Initialized tracker for object", i)


def track_objects(boxes, frame):
	# Update tracker
	for i in range(len(boxes)):
		ok, boxes[i] = trackers[i].update(frame)

		if not ok:
			print("Cannot locate object ", i)
	
	return boxes



# Run the network here
if __name__ == "__main__":

    # Set the network
    print('Instantiating Tiny Yolo Net...')
    yolo = tiny_yolo_net.TinyYoloNet()
    yolo.set_weights('./weights/yolo-tiny.weights')

    # Determine which tracking algorithm to run
    alg = tracker_type[int(FLAGS.alg)]
    print('Using tracking algorithm: ', alg)

    detect_rate = FLAGS.detect_rate

    if FLAGS.video != None:
        print('Processing video...')
        cap = cv2.VideoCapture(FLAGS.video)

        boxes = []
        boxed_frame = None
        frame_num = 0

        # Set Recording parameters
        if FLAGS.record != None:
            fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
            out = cv2.VideoWriter(FLAGS.record, fourcc, 29.0, (488,488))

        while(cap.isOpened()):
            ret, frame = cap.read()
            frame_num = frame_num + 1
            print ("frame: ", frame_num)

            # Break if no more
            if ret == False:
                break

            # Yolo Processing
            # We will be calling the detecter every so often to help alleviate accumulating tracking error
            if frame_num % detect_rate == 0 or frame_num == 1:
                print ("YOLO BITCHES")
                proc_frame = proc_load_image(frame, (448,448))
                boxes = yolo.process(proc_frame, thresholds=[0.17,0.17], classes=[6,14], iou_threshold=0.4)

                init_tracker(boxes, frame, alg)

            else:
                # Tracking processing
                boxes = track_objects(boxes, frame)

            #print ("tracker out: ", boxes)

            # Display
            #boxed_frame, boxes = draw_box(boxes,cv2.resize(frame,(488,488)))
            boxed_frame = draw_tracker_box(boxes, frame)
            #print ("Yolo out: ", boxes)
            cv2.imshow('Yolo Out', boxed_frame)

            # Save to file
            if FLAGS.record != None:
                out.write(boxed_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        print('Finished!')
        cap.release()
        cv2.destroyAllWindows()

    elif FLAGS.image != None:
        print("Processing image")
        # Test
        proc_im = proc_load_image(cv2.imread(FLAGS.image), (448,448))
        im = cv2.resize(cv2.imread(FLAGS.image), (448,448))
        boxes = yolo.process(proc_im, thresholds=[0.1,0.1], classes=[6,14], iou_threshold=0.4)
        boxed_im, boxes = draw_box(boxes,im)
        cv2.imwrite('out.jpg', cv2.cvtColor(boxed_im,cv2.COLOR_BGR2RGB))
    else:
        print('No file specified')

