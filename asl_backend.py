import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import seri
import text_to_speech as mtts

labelFull = ""

# Prepare data generator for standardizing frames before sending them into the model.
data_generator = ImageDataGenerator(samplewise_center=True, samplewise_std_normalization=True)

MODEL_NAME = 'asl_alphabet_codeX.h5'
model = load_model(MODEL_NAME)

# Sizes that used to draw a box which represents the area where the identification part is performed
IMAGE_SIZE = 200
CROP_SIZE = 400

classes_file = open("classes.txt")
classes_string = classes_file.readline()
classes = classes_string.split()
classes.sort()

cap = cv2.VideoCapture(0)

while (True):
    ret, frame = cap.read()

    cv2.rectangle(frame, (0, 0), (CROP_SIZE, CROP_SIZE), (0, 255, 0), 3)

    cropped_image = frame[0:CROP_SIZE, 0:CROP_SIZE]
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
            pass
        elif predicted_class == "del":
            labelFull = labelFull[:-1]
        elif predicted_class == "space":
            if len(labelFull) == 0:
                pass
            else:
                if labelFull[-1] != " ":
                    labelFull += str(" ")
        else:
            if len(labelFull) == 0:
                labelFull += str(predicted_class)
            else:
                if labelFull[-1] != predicted_class:
                    labelFull += str(predicted_class)
    else:
        cv2.putText(frame, classes[-2], (10, 450), 1, 2, (255, 255, 0), 2, cv2.LINE_AA)

    # Printing text
    cv2.putText(frame, labelFull, (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
    # Display the image with prediction.
    cv2.imshow('frame', frame)
    # Press q to quit
    k = cv2.waitKey(1) & 0xFF
    if k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

seri.save_the_text(labelFull)
mtts.run_voice(labelFull)
