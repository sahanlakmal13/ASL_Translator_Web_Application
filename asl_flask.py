import cv2
from imutils.video import WebcamVideoStream
import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

cap = cv2.VideoCapture(0)

class VideoCamera(object):
    def __init__(self):
        self.stream = WebcamVideoStream(src=0).start()

    def __del__(self):
        self.stream.stop()

    def get_frame(self):

        IMAGE_SIZE = 200
        CROP_SIZE = 400

        data_generator = ImageDataGenerator(samplewise_center=True, samplewise_std_normalization=True)

        MODEL_NAME = 'asl_alphabet_codeX.h5'
        model = load_model(MODEL_NAME)

        classes_file = open("classes.txt")
        classes_string = classes_file.readline()
        classes = classes_string.split()
        classes.sort()

        image = self.stream.read()

        cv2.rectangle(image, (0, 0), (CROP_SIZE, CROP_SIZE), (0, 255, 0), 3)
        cropped_image = image[0:CROP_SIZE, 0:CROP_SIZE]

        resized_frame = cv2.resize(cropped_image, (IMAGE_SIZE, IMAGE_SIZE))
        reshaped_frame = (np.array(resized_frame)).reshape((1, IMAGE_SIZE, IMAGE_SIZE, 3))
        frame_for_model = data_generator.standardize(np.float64(reshaped_frame))

        # Predicting the frame.
        prediction = np.array(model.predict(frame_for_model))
        predicted_class = classes[prediction.argmax()]  # Selecting the max confidence index.

        # Preparing output based on the model's confidence.
        prediction_probability = prediction[0, prediction.argmax()]

        full_label = ""
        if prediction_probability > 0.7:
            if predicted_class == "nothing":
                pass
            elif predicted_class == "del":
                full_label = full_label[:-1]
            elif predicted_class == "space":
                if len(full_label) == 0:
                    pass
                else:
                    if full_label[-1] != " ":
                        full_label += str(" ")
            else:
                if len(full_label) == 0:
                    full_label += str(predicted_class)
                else:
                    if full_label[-1] != predicted_class:
                        full_label += str(predicted_class)
        else:
            cv2.putText(image, classes[-2], (10, 450), 1, 2, (255, 255, 0), 2, cv2.LINE_AA)


        cv2.putText(image, full_label, (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
        ret, jpeg = cv2.imencode('.jpg', image)
        data = []
        data.append(jpeg.tobytes())
        return data
