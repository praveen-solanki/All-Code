import os
import numpy as np
from PIL import Image
import cv2
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR,"C:\\Users\\praveen solanki\\Desktop\\photo\\pintu")

face_cascade = cv2.CascadeClassifier('C:\\Users\\praveen solanki\\Desktop\\Open-CV\\cascades\\haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

current_id = 0
label_ids={}
y_label = []
x_train = []

# Grabing the importanta files and dir name for classes
for root,dirs,files in os.walk(image_dir):
    for file in files:
        if file.endswith('jpg') or file.endswith('jpeg'):
            path = os.path.join(root,file)
            label = os.path.basename(root).replace(" ", "_").lower()
            if label in label_ids:
                pass
            else:
                label_ids[label] = current_id
                current_id = current_id + 1
            id = label_ids[label]
            print(label_ids)

            # converting image into numpy array
            pil_image = Image.open(path).convert('L') # grey scale
            image_array = np.array(pil_image,"uint8")

            faces = face_cascade.detectMultiScale(image_array,1.3,5)

            for (x,y,w,h) in faces:
                roi = image_array[y:y+h, x:x+w]
                x_train.append(roi)
                y_label.append(id)

#print(y_label)
#print(x_train)

with open('labels.pickle','wb') as f:
    pickle.dump(label_ids, f)

recognizer.train(x_train,np.array(y_label))
recognizer.save('trained_face.yml')
print('success')