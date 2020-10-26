#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Edward J. C. Ashenbert'
__credits__ = ["Edward J. C. Ashenbert"]
__maintainer__ = "Edward J. C. Ashenbert"

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from smart_camera_common_imports import *
from SmartCameraModel.thermal_analysis import ThermalAnalysis
from SmartCameraModel.face_detect_recognize import FaceDetectAndRecognition
from SmartCameraModel.face_recognizer_training import FaceRecognizerTraining
import smart_camera_share_memory as sc_share_memory
import uuid


class SmartCameraTrainingController:
    def __init__(self):
        self.running = True
        self.id_directory = uuid.uuid1().hex
        self.thermal_analysis = ThermalAnalysis()
        self.face_detect_recognize = FaceDetectAndRecognition()
        self.face_training = FaceRecognizerTraining()
        self.abs_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def threading_streaming(self, camera):
        normal_camera = cv2.VideoCapture(camera)
        while (not normal_camera.isOpened()):
            normal_camera = cv2.VideoCapture(camera)
        while (self.running):
            ret, frame = normal_camera.read()
            if ret:
                frame = frame[101: 381, 173: 560]
                sc_share_memory.frame_face["frame"] = frame
                time.sleep(0.02)

    def threading_collect_data(self):
        count = 0
        while self.running:

            if not os.path.isdir(self.specific_data_path):
                os.makedirs(self.specific_data_path)

            if sc_share_memory.start_collecting:
                if sc_share_memory.frame_face["frame"] is not None:
                    count += 1
                    frame = sc_share_memory.frame_face["frame"]
                    faces, _ = self.face_training.detect_face(frame)
                    for face in faces:
                        cv2.imwrite(os.path.join(self.specific_data_path, str(count) + IMAGE_TAIL), face)
                        count += 1
            else:
                self.id_directory = uuid.uuid1().hex
                self.specific_data_path = os.path.join(self.abs_path, DIRECTORY_DATASET, self.id_directory)

            time.sleep(0.05)


if __name__ == "__main__":
    smart_camera_controller = SmartCameraTrainingController()
    threading.Thread(target=smart_camera_controller.threading_streaming, args=[-1, ]).start()

    while True:

        if sc_share_memory.frame_face["frame"] is not None and sc_share_memory.frame_face["thermal"] is not None:
            print("Frame receive")
            frame = sc_share_memory.frame_face["frame"]
            thermal = sc_share_memory.frame_face["thermal"]
            cv2.imshow("thermal", thermal)
            cv2.imshow("frame", frame)
        else:
            print("No frame receive")
        cv2.waitKey(1)