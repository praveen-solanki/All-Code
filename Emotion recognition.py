import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
from keras.preprocessing.image import load_img, img_to_array
import warnings

warnings.filterwarnings("ignore")

# load model
model = load_model("models/emotion/best_model.h5")
cap = cv2.VideoCapture(0)
faceCascade = cv2.CascadeClassifier(
    "models/cascade file/haarcascade_frontalface_default.xml")
while True:
    # captures frame and returns boolean value and captured image
    ret, test_img = cap.read()
    if not ret:
        continue
    gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
    faces_detected = faceCascade.detectMultiScale(gray_img, 1.32, 5)
    for (x, y, w, h) in faces_detected:
        cv2.rectangle(test_img, (x, y), (x + w, y + h),
                      (255, 0, 0), thickness=7)
        # cropping region of interest i.e. face area from  image
        roi_gray = gray_img[y:y + w, x:x + h]
        roi_gray = cv2.resize(roi_gray, (224, 224))
        img_pixels = image.img_to_array(roi_gray)
        img_pixels = np.expand_dims(img_pixels, axis=0)
        img_pixels /= 255

        predictions = model.predict(img_pixels)

        # find max indexed array
        max_index = np.argmax(predictions[0])

        emotions = ('angry', 'disgust', 'fear', 'happy',
                    'sad', 'surprise', 'neutral')
        predicted_emotion = emotions[max_index]

        cv2.putText(test_img, predicted_emotion, (int(x), int(y)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    resized_img = cv2.resize(test_img, (1000, 700))
    cv2.imshow('show',resized_img)

    if cv2.waitKey(1) == 27:
        break
cap.release()
cv2.destroyAllWindows()