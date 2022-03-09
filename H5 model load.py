import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
from keras.models import load_model
import numpy as np
import warnings
warnings.filterwarnings('ignore')

facedetect = cv2.CascadeClassifier(
    'C:\\Users\\praveen solanki\\Desktop\\Open-CV\\cascades\\haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
font = cv2.FONT_HERSHEY_COMPLEX

model = load_model('my_face_model.h5')
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])


while True:
    sucess, imgOrignal = cap.read()
    faces = facedetect.detectMultiScale(imgOrignal, 1.3, 5)
    for x, y, w, h in faces:
        crop_img = imgOrignal[y:y + h, x:x + h]
        img = cv2.resize(crop_img, (200, 200))
        img = img.reshape(1, 200, 200, 3)
        img = np.array(img)
        #img = np.expand_dims(img, axis=0)
        #img = np.vstack([img])
        prediction = model.predict(img)
        probabilityValue = np.amax(prediction)
        #print(prediction)
        if prediction == 1:
            cv2.rectangle(imgOrignal, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.rectangle(imgOrignal, (x, y - 40), (x + w, y), (0, 255, 0), -2)
            # print(get_className(classIndex))
            cv2.putText(imgOrignal, 'praveen', (x, y - 10), font, 0.75, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(imgOrignal, str(round(probabilityValue * 100, 2)) + "%", (180, 75), font, 0.75, (255, 0, 0), 2,
                        cv2.LINE_AA)
        else:
            cv2.rectangle(imgOrignal, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.rectangle(imgOrignal, (x, y - 40), (x + w, y), (0, 255, 0), -2)
            cv2.putText(imgOrignal, 'unknown', (x, y - 10), font, 0.75, (255, 255, 255), 1, cv2.LINE_AA)
    
            cv2.putText(imgOrignal, str(round(probabilityValue * 100, 2)) + "%", (280, 75), font, 0.75, (255, 0, 0), 2,
                        cv2.LINE_AA)
        cv2.imshow("Result", imgOrignal)
        k = cv2.waitKey(1)
        if k == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()