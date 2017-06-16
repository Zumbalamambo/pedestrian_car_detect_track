import cv2
import sys

tracker_type = ["MIL", "BOOSTING", "MEDIANFLOW", "TLD", "KCF"]

# bboxes     - list of bounding boxes to track
# track_algo - the tracker to use:
#       0 - MIL
#       1 - BOOSTING
#       2 - MEDIANFLOW
#       3 - TLD
#       4 - KCF
def track_objects(bboxes, track_algo):

    algo = tracker_type[track_algo]
    num_objects = len(bboxes)

    print("Tracking ", num_objects, " number of objects using ", algo)

    # List of tracker objects
    trackers = []

    # Create one tracker for each object
    for j in range(num_objects):
        trackers.append(cv2.Tracker_create(algo))

    # Read video
    video = cv2.VideoCapture("videos/new_bball.mp4")

    # Exit if video not opened
    if not video.isOpened():
        print ("Could not open video")
        sys.exit()

    # Read first frame.
    ok, frame = video.read()
    if not ok:
        print ("Cannot read video file")
        sys.exit()

    # Initialize tracker with first frame for each object
    for j in range(num_objects):
        ok = trackers[j].init(frame, bboxes[j])

        if(ok == False):
            print("Couldn't initialize tracker for object ", j)
        else:
            print("Initialized tracker for object", j)

    while True:
        # Read a new frame
        ok_read, frame = video.read()
        if not ok_read:
            break

        # Update tracker
        for j in range(num_objects):
            ok, bboxes[j] = trackers[j].update(frame)

            if ok:
                p1 = (int(bboxes[j][0]), int(bboxes[j][1]))
                p2 = (int(bboxes[j][0] + bboxes[j][2]), int(bboxes[j][1] + bboxes[j][3]))
                cv2.rectangle(frame, p1, p2, (0,0,255), 5)
            else:
                print("Cannot locate object ", j)

        # Display result
        cv2.imshow("Tracking", frame)

        # Exit if ESC pressed
        k = cv2.waitKey(1) & 0xff
        if k == 27 : break
# Comment out for testing
#if __name__ == '__main__' :
#    bboxes = [
#    (747.0, 247.0, 27.0, 133.0),
#    (406.0, 327.0, 33.0, 170.0),
#    (804.0, 397.0, 88.0, 168.0),
#    (1009.0, 222.0, 76.0, 150.0),
#    (532.0, 196.0, 37.0, 158.0)
#    ]
#    print(bboxes)

#    track_objects(bboxes, 4)