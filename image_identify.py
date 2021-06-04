import cv2
import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def snap_feed(picName):
    labelFull = ""

    # Prepare data generator for standardizing frames before sending them into the model.
    data_generator = ImageDataGenerator(samplewise_center=True, samplewise_std_normalization=True)

    MODEL_NAME = 'asl_alphabet_codeX.h5'
    model = load_model(MODEL_NAME)

    IMAGE_SIZE = 200
    CROP_SIZE = 400

    classes_file = open("classes.txt")
    classes_string = classes_file.readline()
    classes = classes_string.split()
    classes.sort()

    image = cv2.imread(picName)

    cropped_image = image[0:CROP_SIZE, 0:CROP_SIZE]
    resized_frame = cv2.resize(cropped_image, (IMAGE_SIZE, IMAGE_SIZE))
    reshaped_frame = (np.array(resized_frame)).reshape((1, IMAGE_SIZE, IMAGE_SIZE, 3))
    frame_for_model = data_generator.standardize(np.float64(reshaped_frame))

    # Predicting the frame.
    prediction = np.array(model.predict(frame_for_model))
    predicted_class = classes[prediction.argmax()]  # Selecting the max confidence index.

    # Preparing output based on the model's confidence.
    prediction_probability = prediction[0, prediction.argmax()]
    if prediction_probability > 0.7:
        if predicted_class == "nothing":
            labelFull += "No Letter Identified"
        elif predicted_class == "space":
            labelFull += "Space"
        else:
            if len(labelFull) == 0:
                labelFull += str(predicted_class)
            else:
                if labelFull[-1] != predicted_class:
                    labelFull += str(predicted_class)
    else:
        labelFull += "No Letter Identified"

    return labelFull;