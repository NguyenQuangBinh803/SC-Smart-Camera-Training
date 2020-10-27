#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Edward J. C. Ashenbert'
__credits__ = ["Edward J. C. Ashenbert"]
__maintainer__ = "Edward J. C. Ashenbert"

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import smart_camera_share_memory as sc_share_memory
from smart_camera_common_imports import *


class OwnImageWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(OwnImageWidget, self).__init__(parent)
        self.image = None

    def setImage(self, image):
        self.image = image
        sz = image.size()
        self.setMinimumSize(sz)
        self.update()

    def paintEvent(self, event):
        qp = QtGui.QPainter()
        qp.begin(self)
        if self.image:
            qp.drawImage(QtCore.QPoint(0, 0), self.image)
        qp.end()

# ABS_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class Window(QtWidgets.QMainWindow, uic.loadUiType(os.path.join(os.path.dirname(os.path.abspath(__file__)), "smart_camera_training_ui.ui"))[0]):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.title = "SMART CAMERA TRAINING SYSTEM"
        self.InitUI()
        self.show()

    def InitUI(self):
        self.setWindowTitle(self.title)

        self.camera = OwnImageWidget(self.camera)

        self.face_image = self.label_5
        self.face_image.setScaledContents(True)

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.face_update_frame)
        self.timer.start(0)

        self.timer_2 = QtCore.QTimer(self)
        self.timer_2.timeout.connect(self.progress_bar_update)
        self.timer_2.start(0)


        self.start_collecting_button = self.pushButton
        self.enter_your_name_button = self.pushButton_2
        self.training_button = self.pushButton_3

        self.start_collecting_button.clicked.connect(self.activate_collect_data)
        self.training_button.clicked.connect(self.activate_training)
        self.enter_your_name_button.clicked.connect(self.activate_enter_name)
        self.textEdit.setAlignment(QtCore.Qt.AlignCenter)


    def activate_collect_data(self):
        sc_share_memory.start_collecting = True

    def activate_training(self):
        sc_share_memory.start_training = True

    def activate_enter_name(self):
        sc_share_memory.target_name = self.textEdit.toPlainText()
        sc_share_memory.target_name_entered = True

    def progress_bar_update(self):
        self.progressBar.setValue(sc_share_memory.collecting_status)

        height, width, channel = sc_share_memory.global_face_image.shape
        bytesPerLine = 3 * width
        qImg = QtGui.QImage(sc_share_memory.global_face_image.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB32)
        pic = QtGui.QPixmap(qImg)
        self.face_image.setPixmap(pic)

    def face_update_frame(self):
        if sc_share_memory.frame_face["frame"] is not None:
            frame = sc_share_memory.frame_face["frame"]
            frame = cv2.resize(frame, (591, 441))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            bpl = frame.shape[2] * frame.shape[1]
            image_in = QtGui.QImage(frame.data, 591, 441, bpl, QtGui.QImage.Format_RGB888)
            self.camera.setImage(image_in)

    def closeEvent(self, event):
        global running
        running = False
        event.accept()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    main_window = Window()
    sys.exit(app.exec_())
