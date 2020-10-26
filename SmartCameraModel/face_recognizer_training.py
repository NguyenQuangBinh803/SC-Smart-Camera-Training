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
import glob

# import face_recognition

from smart_camera_common_imports import *


class FaceRecognizerTraining:

    def __init__(self):
        self.abs_path = os.path.dirname(os.path.abspath(__file__))
        self.embedder = cv2.dnn.readNetFromTorch(os.path.join(self.abs_path, DIRECTORY_EMBEDDING_MODEL, EMBEDDING_MODEL))
        self.face_detector_model = os.path.join(self.abs_path, DIRECTORY_FACE_DETECTOR, DETECTOR_MODEL)
        self.face_detector_weights = os.path.join(self.abs_path, DIRECTORY_FACE_DETECTOR, DETECTOR_WEIGHTS)

        self.faceNet = cv2.dnn.readNetFromCaffe(self.face_detector_model, self.face_detector_weights)


    def encoding_with_torch_openface(self, dataset_directory):
        folders = glob.glob(os.path.join(self.abs_path, dataset_directory) + os.sep + "*" + os.sep)
        knownNames = []
        knownEmbeddings = []

        for folder in folders:
            target_name = os.path.basename(os.path.normpath(folder))
            print(folder)
            images = glob.glob(folder + "*.jpg")
            print(folder, len(images))
            total = 0
            for image in images:
                img = cv2.imread(image)
                start = time.time()
                faceBlob = cv2.dnn.blobFromImage(img, 1.0 / 255,
                                                 (96, 96), (0, 0, 0), swapRB=True, crop=False)
                self.embedder.setInput(faceBlob)
                vec = self.embedder.forward()
                print(target_name, time.time() - start)
                knownNames.append(target_name)
                knownEmbeddings.append(vec.flatten())
                total += 1

        data = {"embeddings": knownEmbeddings, "names": knownNames}
        f = open(os.path.join(self.abs_path, DIRECTORY_EMBEDDING_DATA, EMBEDDING_DATA), "wb")
        f.write(pickle.dumps(data))
        f.close()

    def training_svm_model_openface(self):
        self.embeddings_data = pickle.loads(open(os.path.join(self.abs_path, DIRECTORY_EMBEDDING_DATA,
                                                              EMBEDDING_DATA),
                                                 "rb").read())

        print("[INFO] encoding labels...")
        le = LabelEncoder()
        labels = le.fit_transform(self.embeddings_data["names"])

        print("[INFO] training model...")
        recognizer = SVC(C=1.0, kernel="linear", probability=True)
        recognizer.fit(self.embeddings_data["embeddings"], labels)

        f = open(os.path.join(self.abs_path, DIRECTORY_EMBEDDING_RECOGNIZER, EMBEDDING_RECOGNIZER), "wb")
        f.write(pickle.dumps(recognizer))
        f.close()

        f = open(os.path.join(self.abs_path, DIRECTORY_EMBEDDING_RECOGNIZER, EMBEDDING_LABEL), "wb")
        f.write(pickle.dumps(le))
        f.close()

    def detect_face(self, frame, rgb_require=True):

        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224), (104.0, 177.0, 123.0))
        self.faceNet.setInput(blob)
        detections = self.faceNet.forward()
        faces = []
        locs = []
        preds = []
        rgb_faces = []
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                (startX, startY) = (max(0, startX), max(0, startY))
                (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
                try:
                    face = frame[startY:endY, startX:endX]
                    faces.append(face)
                    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

                except Exception as exp:
                    print(str(exp))
        if rgb_require:
            return locs, rgb_faces
        else:
            return locs, faces

    # def recognize_with_openface(self, face):
    #     recognizer = pickle.loads(open(os.path.join(self.abs_path, DIRECTORY_EMBEDDING_DATA, EMBEDDING_RECOGNIZER),
    #     "rb").read())
    #     le = pickle.loads(open(os.path.join(self.abs_path, DIRECTORY_EMBEDDING_DATA, EMBEDDING_LABEL), "rb").read())
    #
    #     faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
    #     self.embedder.setInput(faceBlob)
    #     vec = self.embedder.forward()
    #
    #     # perform classification to recognize the face
    #     preds = recognizer.predict_proba(vec)[0]
    #     j = np.argmax(preds)
    #     proba = preds[j]
    #     name = le.classes_[j]


if __name__ == "__main__":
    face_encode = FaceRecognizerTraining()
    face_encode.encoding_with_torch_openface("dataset")
    face_encode.training_svm_model_openface()
