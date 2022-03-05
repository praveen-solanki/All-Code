import cv2
import numpy as np
import face_recognition
import os


class Face():
    def __init__(self):
        self.run()
    def run(self):
        self.path = 'C:\\Users\\praveen solanki\\Desktop\\photo\\pintu'
        images = []
        className = []
        self.myList = os.listdir(self.path)
        for c1 in self.myList:
            curImg = cv2.imread(f'{self.path}/{c1}')
            images.append(curImg)
            className.append(os.path.splitext(c1)[0])
        self.encodeList = []
        for img in images:
            self.img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.encode = face_recognition.face_encodings(self.img)[0]
            self.encodeList.append(self.encode)
        self.encodeListKnown = self.encodeList
        self.cap = cv2.VideoCapture(0)

        while True:
            ret, frame = self.cap.read()
            img1 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            self.faceFrame = face_recognition.face_locations(img1)
            self.encodeFrame = face_recognition.face_encodings(img1, self.faceFrame)
            for encodeFace, faceLoc in zip(self.encodeFrame, self.faceFrame):
                self.matches = face_recognition.compare_faces(self.encodeListKnown, encodeFace)
                self.faceDis = face_recognition.face_distance(self.encodeListKnown, encodeFace)
                matchIndex = np.argmin(self.faceDis)

                if self.matches[matchIndex]:
                    name = className[matchIndex].upper()
                    y1, x2, y2, x1 = faceLoc
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.rectangle(frame, (x1, y2-30), (x2, y2),(0, 255, 0), cv2.FILLED)
                    cv2.putText(frame, name, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

                else:
                    y1, x2, y2, x1 = faceLoc
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.rectangle(frame, (x1, y2-30), (x2, y2),(0, 255, 0), cv2.FILLED)
                    cv2.putText(frame, 'UNKNOWN', (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

            cv2.imshow('video', frame)
            if cv2.waitKey(24) == 27:
                break

        self.cap.release()                
        cv2.destroyAllWindows

# if __init__ == "__main__":
Face()
