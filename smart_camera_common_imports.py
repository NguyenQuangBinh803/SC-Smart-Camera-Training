from tensorflow import keras
import tensorflow as tf
import cv2
import time
from datetime import datetime
from PyQt5 import uic

from PyQt5 import QtCore, QtGui, QtWidgets
import threading
import imutils

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

import math
import numpy as np
import pickle
import os

from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import argparse


# prototxtPath = os.path.abspath("SmartCameraModel/face_detector/deploy.prototxt")
# weightsPath = os.path.abspath("SmartCameraModel/face_detector/res10_300x300_ssd_iter_140000.caffemodel")
# MaskModelTFlitePath = os.path.abspath("SmartCameraModel/model.tflite")
# EncodingDataPath = os.path.abspath("SmartCameraModel/embedding_data/encodings.pickle")
# CameraUIPath = os.path.abspath("SmartCameraView/smart_camera_training_ui.ui")
# IconPath = os.path.abspath("SmartCameraView/")
# FaceImagePath = os.path.abspath("SmartCameraModel/")

DIRECTORY_EMBEDDING_DATA = "embedding_data"
DIRECTORY_EMBEDDING_RECOGNIZER = "embedding_recognizer"
DIRECTORY_EMBEDDING_MODEL = "embedding_model"
DIRECTORY_FACE = "face_images"
DIRECTORY_FACE_DETECTOR = "face_detector"
DIRECTORY_MASK_DETECTOR = "mask_detector_model"
DIRECTORY_DATASET = "dataset"

MASK_MODEL = "model.tflite"
DETECTOR_MODEL = "deploy.prototxt"
DETECTOR_WEIGHTS = "res10_300x300_ssd_iter_140000.caffemodel"
EMBEDDING_DATA = "embeddings.pickle"
EMBEDDING_RECOGNIZER = "recognizer.pickle"
EMBEDDING_MODEL = "openface_nn4.small2.v1.t7"
EMBEDDING_LABEL = "le.pickle"


DATE_AND_TIME = str(datetime.now().strftime("%Y%m%d"))
IMAGE_TAIL = ".jpg"



FACE_RECOGNIZE_THRESHOLD = 0.8
FACE_AREA_THRESHOLD = 3000