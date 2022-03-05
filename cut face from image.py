import cv2
import os

#cap = cv2.VideoCapture(0)
path = 'myimage/training/not_me/'

cap = cv2.VideoCapture('C:\\Users\\praveen solanki\\Downloads\\Faces from around the world (1).mp4')
count = 0
face_cascade = cv2.CascadeClassifier('C:\\Users\\praveen solanki\\Desktop\\Open-CV\\cascades\\haarcascade_frontalface_default.xml')

while True:
    ret, frame = cap.read()
    video = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(video, 1.3, 5)
    for (x,y,w,h) in faces:
        #cv2.rectangle(video, (x, y), (x+w, y+h), (0,0,255), 2)
        roi_gray = video[y:y+h+30, x:x+w]
        cv2.imshow('video',roi_gray)

        name = path + 'praveen-'+str(count) +'.jpg'
        cv2.imwrite(name,roi_gray)
        count = count + 1
        print(count)
    if count == 2000 :
        print(f"Sucessfully captured {count}")
        break
    if cv2.waitKey(24) == 27:
        break
cap.release()
cv2.destroyAllWindows()