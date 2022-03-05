import numpy as np
import face_recognition
import os
import cv2

path = 'C:\\Users\\praveen solanki\\Desktop\\photo\\pintu'
images = []
className = []
myList = os.listdir(path)

for c1 in myList:
    curImg = cv2.imread(f'{path}/{c1}')
    images.append(curImg)
    className.append(os.path.splitext(c1)[0])

def findEncoding(images):
    encodeList = []
    for img in images:
        #print(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
        #print(encodeList)
    return encodeList

encodeListKnown = findEncoding(images)
print('Please Wait!')
cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()
    img1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    faceFrame = face_recognition.face_locations(img1)
    encodeFrame = face_recognition.face_encodings(img1, faceFrame)
    print(encodeFrame)
    for encodeFace, faceLoc in zip(encodeFrame, faceFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = className[matchIndex].upper()
            y1, x2, y2, x1 = faceLoc
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2-30), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1+6, y2-6),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
        else:
            y1, x2, y2, x1 = faceLoc
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2-30), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, 'UNKNOWN', (x1+6, y2-6),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('video',img)
    if cv2.waitKey(24) == 27:
        break
    cv2.destroyAllWindows