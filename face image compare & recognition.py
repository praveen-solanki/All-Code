import cv2
import numpy as np
import face_recognition

img1 = face_recognition.load_image_file('images/praveen.jpg')
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

img2 = face_recognition.load_image_file('images/YASH.jpeg')
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

faceLoc1 = face_recognition.face_locations(img1)[0]
encode1 = face_recognition.face_encodings(img1)[0]
cv2.rectangle(img1,(faceLoc1[3],faceLoc1[0]),(faceLoc1[1],faceLoc1[2]),(0,255,0))

faceLoc2 = face_recognition.face_locations(img2)[0]
encode2 = face_recognition.face_encodings(img2)[0]
cv2.rectangle(img2,(faceLoc2[3],faceLoc2[0]),(faceLoc2[1],faceLoc2[2]),(0,255,0))


result = face_recognition.compare_faces([encode1],encode2)
distance = face_recognition.face_distance([encode1],encode2)
print(result,distance)


cv2.imshow('elon image', img1)
cv2.imshow('elon image 2', img2)
cv2.waitKey(0)