#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Edward J. C. Ashenbert'
__credits__ = ["Edward J. C. Ashenbert"]
__maintainer__ = "Edward J. C. Ashenbert"

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from SmartCameraView.smart_camera_training_ui import *
from SmartCameraController.smart_camera_training_controller import SmartCameraTrainingController

if __name__ == "__main__":
    smart_camera_training_controller = SmartCameraTrainingController()
    app = QtWidgets.QApplication(sys.argv)
    threading.Thread(target=smart_camera_training_controller.threading_streaming, args=[-1, ]).start()
    threading.Thread(target=smart_camera_training_controller.threading_collect_data).start()

    main_window = Window()
    sys.exit(app.exec_())