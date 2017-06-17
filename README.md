# Pedestrian And Vehicle Multi Tracking
![dashcam](/images/out.jpg)

# Features
* Tiny YoloNet Detection and Recognition
* OpenCV Tracking

# How to Run
Before running code, a file with the YOLO net weights must be downloaded and
placed into the weights directory.

To run the code:
python main.py --video /path/to/video

Settings can be adjusted using the following flags and parameters:
            FLAG                     parameter
----------------------------------------------------------------------
            --video        [video file]
            --image        [image file]
            --record       [file to output recorded yolo]
            --alg          [tracking algorithm to run]
                                     0 - MIL
                                     1 - BOOSTING
                                     2 - MEDIANFLOW
                                     3 - TLD
                                     4 - KCF
            --detect_rate  [# of frames before rerunning detection]
            --tracking     [0 - enable tracking or 1 - disable tracking]

The weights along with sample videos can be downloaded from the following [link](https://drive.google.com/drive/folders/0B8nucpT0jfADM19lOC1rbTVnREE?usp=sharing)

# Dependencies
* Keras 1.2.0
* OpenCV 3.2.0 (with opencv_contrib)
* Tensorflow 1.1.0
* Python 3.4.3

# References:
* https://github.com/xslittlegrass/CarND-Vehicle-Detection
* https://github.com/thtrieu/darkflow
* https://github.com/sunshineatnoon/Darknet.keras/
* https://github.com/allanzelener/YAD2K
