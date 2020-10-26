__author__ = 'Edward J. C. Ashenbert'
__credits__ = ["Edward J. C. Ashenbert"]
__maintainer__ = "Edward J. C. Ashenbert"
__email__ = "nguyenquangbinh803@gmail.com"
__copyright__ = "Copyright 2020"
__status__ = "Working on embedding recognition to smart camera"
__version__ = "1.0.1"

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# import face_recognition
import smart_camera_share_memory as sc_share_memory
from smart_camera_common_imports import *
from tensorflow import keras
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array


class FaceDetectAndRecognition():

    def __init__(self):

        print("Init models")
        self.abs_path = os.path.dirname(os.path.abspath(__file__))
        self.openface_recognizer = pickle.loads(open(os.path.join(self.abs_path, DIRECTORY_EMBEDDING_RECOGNIZER, EMBEDDING_RECOGNIZER), "rb").read())
        self.openface_le = pickle.loads(open(os.path.join(self.abs_path, DIRECTORY_EMBEDDING_RECOGNIZER, EMBEDDING_LABEL), "rb").read())
        self.embedder = cv2.dnn.readNetFromTorch(os.path.join(self.abs_path, DIRECTORY_EMBEDDING_MODEL, EMBEDDING_MODEL))

        # Initialize resnet face detector
        self.face_detector_model = os.path.join(self.abs_path, DIRECTORY_FACE_DETECTOR, DETECTOR_MODEL)
        self.face_detector_weights = os.path.join(self.abs_path, DIRECTORY_FACE_DETECTOR, DETECTOR_WEIGHTS)

        self.faceNet = cv2.dnn.readNetFromCaffe(self.face_detector_model, self.face_detector_weights)
        self.config = tf.ConfigProto(
            device_count={'GPU': 1},
            intra_op_parallelism_threads=1,
            allow_soft_placement=True
        )

        # Initialize keras mask detector
        self.mask_detector_model = os.path.join(self.abs_path, DIRECTORY_MASK_DETECTOR, MASK_MODEL)
        self.session = tf.Session(config=self.config)
        keras.backend.set_session(self.session)
        self.maskNet = tf.lite.Interpreter(model_path=self.mask_detector_model)
        self.maskNet.allocate_tensors()

        self.input_details = self.maskNet.get_input_details()
        self.output_details = self.maskNet.get_output_details()
        self.input_shape = self.input_details[0]['shape']

        print("Done init models")

    def diagnostic_logging(self, message):
        print(message)

    def detect_mask(self, faces_image):
        is_mask = False
        try:
            with self.session.as_default():
                with self.session.graph.as_default():
                    faces = np.array(faces_image, dtype="float32")
                    self.maskNet.set_tensor(self.input_details[0]['index'], faces)
                    self.maskNet.invoke()
                    preds = self.maskNet.get_tensor(self.output_details[0]['index'])
                    try:
                        if len(preds[0]) > 1:
                            if preds[0][1] > 0.5:
                                is_mask = False
                                self.mask_detect_status = False
                            elif preds[0][0] > 0.5:
                                is_mask = True
                                self.mask_detect_status = True
                    except Exception as exp:
                        print(str(exp))

        except Exception as exp:
            print(exp)

        return is_mask

    def face_and_mask_detect(self, frame):
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224), (104.0, 177.0, 123.0))
        self.faceNet.setInput(blob)
        detections = self.faceNet.forward()
        faces = []
        locs = []
        preds = []
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                sc_share_memory.human_appear_status = True
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                (startX, startY) = (max(0, startX), max(0, startY))
                (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
                face = frame[startY:endY, startX:endX]
                sc_share_memory.global_face_image = face
                cv2.imwrite(os.path.join(self.abs_path, DIRECTORY_FACE, DATE_AND_TIME + IMAGE_TAIL), face)
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (224, 224))
                face = img_to_array(face)
                face = preprocess_input(face)
                faces.append(face)

                sc_share_memory.face_area = (endX - startX) * (endY - startY)

                if (endX - startX) * (endY - startY) > 3000:
                    sc_share_memory.face_detect_status = True
                else:
                    sc_share_memory.face_detect_status = False

                locs.append((startX, startY, endX, endY))
        if not locs:
            sc_share_memory.face_detect_status = False
            sc_share_memory.human_appear_status = False

        if sc_share_memory.face_detect_status:
            sc_share_memory.global_locs = locs

        if len(faces) > 0:
            try:
                with self.session.as_default():
                    with self.session.graph.as_default():
                        faces = np.array(faces, dtype="float32")
                        self.maskNet.set_tensor(self.input_details[0]['index'], faces)
                        self.maskNet.invoke()
                        preds = self.maskNet.get_tensor(self.output_details[0]['index'])
                        print(preds)
                        try:
                            if len(preds[0]) > 1:
                                if preds[0][1] > 0.5:
                                    is_mask = False
                                    sc_share_memory.mask_detect_status = False
                                elif preds[0][0] > 0.5:
                                    is_mask = True
                                    sc_share_memory.mask_detect_status = True
                        except Exception as exp:
                            print(str(exp))

            except Exception as exp:
                print(exp)
        return (locs, preds)

    def face_recognize_openface(self, face):
        faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                                         (96, 96), (0, 0, 0), swapRB=True, crop=False)
        self.embedder.setInput(faceBlob)
        vec = self.embedder.forward()
        preds = self.openface_recognizer.predict_proba(vec)[0]

        j = np.argmax(preds)
        proba = preds[j]
        name = self.openface_le.classes_[j]
        print(name)
        if proba > FACE_RECOGNIZE_THRESHOLD:
            return name
        else:
            return "Unknown"


if __name__ == "__main__":
    face_detect = FaceDetectAndRecognition()
    # face_detect.encoding_with_dlib_face("dataset")
    # cap = cv2.VideoCapture(-1)
    # encoding_data = pickle.loads(open("encodings.pickle", "rb").read())
    # id_directory = uuid.uuid1().hex
    # dataset_directory = "dataset/"
    # os.makedirs("dataset/" + id_directory)
    # count = 0
    #
    # while True:
    #     ret, frame = cap.read()
    #
    #     if ret:
    #         locs, face_images = face_detect.detect_face(frame)
    #         for index, face in enumerate(face_images):
    #             count += 1
    #             write_face = frame[locs[index][1]:locs[index][3], locs[index][0]:locs[index][2]]
    #             cv2.imwrite(dataset_directory + id_directory + "/" + str(count) + ".jpg", write_face)
    #             cv2.imshow("Recording", face)
    #             cv2.waitKey(1)
    # face_detect.face_recognize_with_dlib(frame, encoding_data)
