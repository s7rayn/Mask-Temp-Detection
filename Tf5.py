######## Webcam Object Detection Using Tensorflow-trained Classifier #########
#
# Author: Evan Juras
# Date: 10/27/19
# Description: 
# This program uses a TensorFlow Lite model to perform object detection on a live webcam
# feed. It draws boxes and scores around the objects of interest in each frame from the
# webcam. To improve FPS, the webcam object runs in a separate thread from the main program.
# This script will work with either a Picamera or regular USB webcam.
#
# This code is based off the TensorFlow Lite image classification example at:
#	 https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/examples/python/label_image.py
#
# I added my own method of drawing boxes and labels using OpenCV.

# Import packages
import os
import argparse
import cv2
import numpy as np
import sys
import time
from threading import Thread
import importlib.util
# from smbus2 import SMBus
import json
import serial
# from mlx90614 import MLX90614
import threading

global dst
# Define VideoStream class to handle streaming of video from webcam in separate processing thread
# Source - Adrian Rosebrock, PyImageSearch: https://www.pyimagesearch.com/2015/12/28/increasing-raspberry-pi-fps-with-python-and-opencv/
m_t_k = cv2.imread('m_t_k.jpg')
m_k_t_n = cv2.imread('m_k_t_n.jpg')
m_n_t_k = cv2.imread('m_n_t_k.jpg')
m_n_t_n = cv2.imread('m_n_t_n.jpg')


def draw_circle(image, center_coordinates, radius, color, thickness):
    cv2.circle(image, center_coordinates, radius, color, thickness)


def draw_border(img, pt1, pt2, color, thickness, r):
    x1, y1 = pt1
    x2, y2 = pt2

    # Top left
    cv2.line(img, (x1 + r, y1), (x1 + r + 436, y1), color, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y1 + r + 174), color, thickness)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)

    # Top right
    cv2.line(img, (x2 - r, y1), (x2 - r, y1), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y1 + r), color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)

    # Bottom left
    cv2.line(img, (x1 + r, y2), (x1 + r, y2), color, thickness)
    cv2.line(img, (x1, y2 - r), (x1, y2 - r), color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)

    # Bottom right
    cv2.line(img, (x2 - r, y2), (x2 - r - 436, y2), color, thickness)
    cv2.line(img, (x2, y2 - r), (x2, y2 - r - 174), color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)


class VideoStream:
    """Camera object that controls video streaming from the Picamera"""

    def __init__(self, resolution=(480, 640), framerate=30):
        # Initialize the PiCamera and the camera image stream
        self.stream = cv2.VideoCapture(0)
        # print(self.stream.get(cv2.CAP_PROP_FRAME_WIDTH))
        ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        ret = self.stream.set(3, resolution[0])
        ret = self.stream.set(4, resolution[1])

        # Read first frame from the stream
        (self.grabbed, self.frame) = self.stream.read()

        # Variable to control when the camera is stopped
        self.stopped = False

    def start(self):
        # Start the thread that reads frames from the video stream
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        # Keep looping indefinitely until the thread is stopped
        while True:
            # If the camera is stopped, stop the thread
            if self.stopped:
                # Close camera resources
                self.stream.release()
                return

            # Otherwise, grab the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        # Return the most recent frame
        return self.frame

    def stop(self):
        # Indicate that the camera and thread should be stopped
        self.stopped = True


# Define and parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--modeldir', help='Folder the .tflite file is located in',
                    required=True)
parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite',
                    default='detect.tflite')
parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt',
                    default='labelmap.txt')
parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
                    default=0.5)
parser.add_argument('--resolution',
                    help='Desired webcam resolution in WxH. If the webcam does not support the resolution entered, errors may occur.',
                    default='480x640')
parser.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection',
                    action='store_true')

args = parser.parse_args()

MODEL_NAME = args.modeldir
GRAPH_NAME = args.graph
LABELMAP_NAME = args.labels
min_conf_threshold = float(args.threshold)
resW, resH = args.resolution.split('x')
imW, imH = int(resW), int(resH)
use_TPU = args.edgetpu

# Import TensorFlow libraries
# If tflite_runtime is installed, import interpreter from tflite_runtime, else import from regular tensorflow
# If using Coral Edge TPU, import the load_delegate library
pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter

    if use_TPU:
        from tflite_runtime.interpreter import load_delegate
else:
    from tensorflow.lite.python.interpreter import Interpreter

    if use_TPU:
        from tensorflow.lite.python.interpreter import load_delegate

# If using Edge TPU, assign filename for Edge TPU model
if use_TPU:
    # If user has specified the name of the .tflite file, use that name, otherwise use default 'edgetpu.tflite'
    if (GRAPH_NAME == 'detect.tflite'):
        GRAPH_NAME = 'edgetpu.tflite'

    # Get path to current working directory
CWD_PATH = os.getcwd()

# Path to .tflite file, which contains the model that is used for object detection
PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, GRAPH_NAME)

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH, MODEL_NAME, LABELMAP_NAME)

# Load the label map
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Have to do a weird fix for label map if using the COCO "starter model" from
# https://www.tensorflow.org/lite/models/object_detection/overview
# First label is '???', which has to be removed.
if labels[0] == '???':
    del (labels[0])

# Load the Tensorflow Lite model.
# If using Edge TPU, use special load_delegate argument
if use_TPU:
    interpreter = Interpreter(model_path=PATH_TO_CKPT,
                              experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
    print(PATH_TO_CKPT)
else:
    interpreter = Interpreter(model_path=PATH_TO_CKPT)

interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5

# Initialize video stream
videostream = VideoStream(resolution=(imW, imH), framerate=30).start()
time.sleep(1)
previous = ""
temp_normal = False
detected = False
object_name = ""

frame = None
frame1 = None
fr = None


def gfg():
    global previous
    previous = ""


# def detect_frame():
#     while True:
#         frame = videostream.read()
#         frame = cv2.flip(frame, 1)
#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         frame_resized = cv2.resize(frame_rgb, (width, height))
#         detection_process(frame_resized)

def temp_dete():
    # while True:
    # ser.write(b"Hello from Raspberry Pi!\n")
    while True:
        global fr
        global temp_normal
        global ser
        line = ser.readline().decode('utf-8').rstrip()
        if line:
            json_obj = json.loads(line)
            fr = json_obj["ObjectTemp"]
            if fr > 37:
                temp_normal = False
            else:
                temp_normal = True
            # time.sleep(1)
            # return json_obj["ObjectTemp"]


# def temp_dete():
#    global fr
#    global temp_normal
#    bus = SMBus(1)
#    sensor = MLX90614(bus, address=0x5A)
#    Celsius = sensor.get_object_1()
#    fr = ( Celsius * 9/5 ) + 32
#    print( "Temp: " + str(fr) )
#    bus.close()
#
#    if (fr > 98.6):
#        temp_normal = False
#    else:
#        temp_normal = True

def detection_process(frame_resized):
    global frame1
    global previous
    global temp_normal
    global object_name
    # global detected
    input_data = np.expand_dims(frame_resized, axis=0)
    # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std
        # Perform the actual detection by running the model with the image as input
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    # Retrieve detection results
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]  # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[1]['index'])[0]  # Class index of detected objects
    scores = interpreter.get_tensor(output_details[2]['index'])[0]  # Confidence of detected objects
    num = interpreter.get_tensor(output_details[3]['index'])[
        0]  # Total number of detected objects (inaccurate and not needed)
    # Loop over all detections and draw detection box if confidence is above minimum threshol
    for i in range(len(scores)):
        # print(scores[0])
        object_name = ""
        if ((scores[0] > 0.95) and (scores[0] <= 1.0)):
            # Get bounding box coordinates and draw box
            # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
            ymin = int(max(1, (boxes[0][0] * imH)))
            xmin = int(max(1, (boxes[0][1] * imW)))
            ymax = int(min(imH, (boxes[0][2] * imH)))
            xmax = int(min(imW, (boxes[0][3] * imW)))
            # Draw label
            # print(xmin)
            # print(ymin)
            # print(xmax)
            # print(ymax)
            # 100; 100; 500; 780
            if (xmin < 100 and ymin < 100 and xmax > 365 and ymax > 530):
                detected = True
                # print(classes[0])
                object_name = labels[int(classes[0])]  # Look up object name from "labels" array using class index
                # label = '%s %s' % (object_name, "Detected") #Example: 'person: 72%'
                # if object_name == "mask":
                #    draw_border(frame1,(50,50),(550,820),(34,139,34),5,30)
                #    cv2.putText(frame1, label.title(), (160, 80), cv2.FONT_HERSHEY_PLAIN, 2.5, (34,139,34), 2)
                #    if not object_name in previous:
                #        previous = object_name
                #        temp_dete()
                #        timer = threading.Timer(5.0, gfg)
                #        timer.start()
                #    if temp_normal:
                #        labelt = "Temeparature Normal"
                #        cv2.putText(frame1, labelt.title(), (135, 110), cv2.FONT_HERSHEY_PLAIN, 2, (34,139,34), 2)
                #    else:
                #        labelt = "Abnormal Temperature"
                #        cv2.putText(frame1, labelt.title(), (130, 110), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2)
                # if object_name == "no mask":
                #    draw_border(frame1,(50,50),(550,820),(0,0,255),5,30)
                #    cv2.putText(frame1, label.title(), (110, 80), cv2.FONT_HERSHEY_PLAIN, 2.5, (0,0,255), 2)
    # cv2.namedWindow("Object detector", cv2.WND_PROP_FULLSCREEN)
    # cv2.setWindowProperty("Object detector", cv2.WND_PROP_FULLSCREEN, 1)
    # cv2.imshow('Object detector', frame1)
    # if cv2.waitKey(1) == ord('q'):
    #    pass


def detect_frame():
    while True:
        frame = videostream.read()
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (width, height))
        detection_process(frame_resized)
        # time.sleep(1)


fr1 = videostream.read()
fr1 = cv2.flip(fr1, 1)
# print(fr1.shape)

dstII = cv2.imread('dst.jpg')


# print(dstII.shape)


def show_frame():
    global dst
    global frame1
    global object_name
    global previous
    global fr
    global m_n_t_n
    global m_k_t_n
    global m_n_t_k
    global m_t_k
    dstIm = cv2.imread('dst.jpg')
    while True:
        # Grab frame from video stream
        frame1 = videostream.read()
        frame1 = cv2.flip(frame1, 1)
        # temp_dete()
        # cv2.putText(frame1,'TEMP: {0:.2f}'.format(fr),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)
        # draw_border(frame1,(50,50),(550,820),(255,255,255),5,30)
        draw_circle(frame1, (320, 240), 350, (255, 255, 255), 4)
        dstIm = cv2.resize(dstIm, (frame1.shape[1], frame1.shape[0]))
        # frame1 = cv2.resize(frame1, (dstIm.shape[1], dstIm.shape[0]))
        # print(frame1.shape)
        # print(dstIm.shape)
        dst = cv2.addWeighted(frame1, 0.5, dstIm, 0.2, 0)
        # print(frame1.shape)
        # print(dstIm.shape)
        label = '%s %s' % (object_name, "Detected")  # Example: 'person: 72%'
        # print(object_name)
        if object_name == "mask":
            # print("mask")
            draw_circle(frame1, (320, 240), 350, (34, 139, 34), 4)
            # draw_border(frame1,(50,50),(550,820),(34,139,34),5,30)
            cv2.putText(frame1, label.title(), (160, 850), cv2.FONT_HERSHEY_PLAIN, 2.5, (34, 139, 34), 2)
            if not object_name in previous:
                previous = object_name
                # temp_dete()
                timer = threading.Timer(5.0, gfg)
                timer.start()
            if temp_normal:
                labelt = "Temeparature Normal"
                cv2.putText(frame1, labelt.title(), (135, 880), cv2.FONT_HERSHEY_PLAIN, 2, (34, 139, 34), 2)
                cv2.putText(frame1, '{0:.2f}'.format(fr), (250, 910), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2,
                            cv2.LINE_AA)
                m_t_k = cv2.resize(m_t_k, (frame1.shape[1], frame1.shape[0]))
                dst = cv2.addWeighted(frame1, 0.5, m_t_k, 0.2, 0)

            else:
                # 880
                # labelt = "Abnormal Temperature"
                # cv2.putText(frame1, labelt.title(), (130, 180), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2)
                cv2.putText(frame1, "S MASKOU", (230, 180), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
                # cv2.putText(frame1,'{0:.2f}'.format(fr),(250,910),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)
                # m_k_t_n = cv2.resize(m_k_t_n, (frame1.shape[1], frame1.shape[0]))
                # dst = cv2.addWeighted(frame1,0.5,m_k_t_n,0.2,0)
                dst = cv2.addWeighted(frame1, 0.5, dst, 0.2, 0)

        if object_name == "no mask":
            # print("no mask")
            draw_circle(frame1, (320, 240), 350, (0, 0, 255), 4)
            # draw_border(frame1,(50,50),(550,820),(0,0,255),5,30)
            cv2.putText(frame1, label.title(), (110, 850), cv2.FONT_HERSHEY_PLAIN, 2.5, (0, 0, 255), 2)
            if not object_name in previous:
                previous = object_name
                # temp_dete()
                timer = threading.Timer(5.0, gfg)
                timer.start()

            if temp_normal:
                labelt = "Temeparature Normal"
                cv2.putText(frame1, labelt.title(), (135, 336), cv2.FONT_HERSHEY_PLAIN, 2, (34, 139, 34), 2)
                cv2.putText(frame1, '{0:.2f}'.format(fr), (250, 366), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2,
                            cv2.LINE_AA)
                m_n_t_k = cv2.resize(m_n_t_k, (frame1.shape[1], frame1.shape[0]))
                dst = cv2.addWeighted(frame1, 0.5, m_n_t_k, 0.2, 0)
            else:
                labelt = "Abnormal Temperature"
                # 130, 880
                cv2.putText(frame1, labelt.title(), (130, 180), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2)
                cv2.putText(frame1, "BEZ MASKY", (230, 180), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
                cv2.putText(frame1,'{0:.2f}'.format(fr),(250,220),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)
                m_n_t_n = cv2.resize(m_n_t_n, (frame1.shape[1], frame1.shape[0]))
                # dst = cv2.addWeighted(frame1,0.5,m_n_t_n,0.2,0)
                dst = cv2.addWeighted(frame1, 0.5, dst, 0.2, 0)

        # cv2.namedWindow("Object detector", cv2.WND_PROP_FULLSCREEN)
        # cv2.setWindowProperty("Object detector", cv2.WND_PROP_FULLSCREEN, 1)
        # dst = cv2.addWeighted(frame1,0.5,m_t_k,0.2,0)
        cv2.imshow('Object detector', dst)
        if cv2.waitKey(1) == ord('q'):
            break


# show = Thread(target=show_frame)
det_frame = Thread(target=detect_frame)
# show.start()
det_frame.start()

ser = serial.Serial('/dev/ttyUSB0', 9600, timeout=1)
ser.reset_input_buffer()

temp = Thread(target=temp_dete).start()
show_frame()
# Clean up
cv2.destroyAllWindows()
videostream.stop()
