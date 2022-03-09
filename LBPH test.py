import numpy as np
import cv2
import pickle

face_cascade = cv2.CascadeClassifier('C:\\Users\\praveen solanki\\Desktop\\Open-CV\\cascades\\haarcascade_frontalface_default.xml')

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trained_face_data.yml")

labels = {"person_name": 1}
with open("labels.pickle", 'rb') as f:
	og_labels = pickle.load(f)
	labels = {v:k for k,v in og_labels.items()}
	print(labels)

cap = cv2.VideoCapture(0)
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
    	roi_gray = gray[y:y+h, x:x+w] 
    	#roi_color = frame[y:y+h, x:x+w]
    	cv2.rectangle(frame, (x, y), (x+w, y+h), (0,0,255), 2)
    	id, conf = recognizer.predict(roi_gray)
    	if conf>=55 and conf <= 80:
    		#print(id)
    		#print(labels[id])
    		font = cv2.FONT_HERSHEY_SIMPLEX
    		name = labels[id]
    		color = (255, 255, 255)
    		stroke = 2
    		cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)
		
    	#cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv2.imshow('frame',frame)
    if cv2.waitKey(24) == 27:
        break
cap.release()
cv2.destroyAllWindows()