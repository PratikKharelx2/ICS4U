import tensorflow.keras
import numpy as np
import cv2


class main_func_classifier:

    def __init__(self, modelPath, labelsPath=None):
        self.model_path = modelPath
        np.set_printoptions(suppress=False)
        self.model = tensorflow.keras.models.load_model(self.model_path)
        self.data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        self.labels_path = labelsPath
        if self.labels_path:
            label_file = open(self.labels_path, "r")
            self.list_labels = []
            for line in label_file:
                stripped_line = line.strip()
                self.list_labels.append(stripped_line)
            label_file.close()
        else:
            print("No Labels Found")

    def getPrediction(self, img, draw= True, pos=(50, 50), scale=2, color = (0,255,0)):
        image_array = np.asarray(img)
        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
        self.data[0] = normalized_image_array
        prediction = self.model.predict(self.data)
        indexVal = np.argmax(prediction)

        if draw and self.labels_path:
            cv2.putText(img, str(self.list_labels[indexVal]),
                        pos, cv2.FONT_HERSHEY_COMPLEX, scale, color, 2)

        return list(prediction[0]), indexVal