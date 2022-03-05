import tkinter as tk
import cv2
import PIL.Image
import PIL.ImageTk
import os
import face_recognition
import numpy as np
#import pyautogui
#import time

class Window:
    def __init__(self):
        self.root = tk.Tk()
        self.c_height = 320
        self.c_width = 300
        self.root.title('Hello')
        self.root.overrideredirect(True)
        self.root.geometry(f"{self.c_width}x{self.c_height}+{580}+{10}")
        self.root.resizable(False, False)
        self.root.configure(bg='black')
        self.T = tk.Label(self.root, text='RECOGNIZING',
                          font=('times', 15, 'bold'), bg='black', fg='white')
        self.T.pack(fill=tk.BOTH, expand=1)

        self.T1 = tk.Button(self.root, text='Quit',
                          font=('times', 15, 'bold'), bg='black', fg='white',command = self.root.destroy)
        self.T1.pack(fill=tk.BOTH, expand=1,side=tk.BOTTOM)
        self.canvas = tk.Canvas(
            self.root, width=self.c_width, height=self.c_height, bg='gray')
        self.canvas.pack(padx=5, pady=5)
        
        self.video()
        self.root.mainloop()

    def click(self):
        print('text')
        print(self.text.config())

    def video(self):
        '''
        self.path = 'C:\\Users\\praveen solanki\\Desktop\\photo\\pintu'
        images = []
        className = []
        self.myList = os.listdir(self.path)
        for c1 in self.myList:
            self.curframe = cv2.imread(f'{self.path}/{c1}')
            images.append(self.curframe)
            className.append(os.path.splitext(c1)[0])
        self.encodeList = []
        for img in images:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.encode = face_recognition.face_encodings(img)[0]
            self.encodeList.append(self.encode)
        self.encodeListKnown = self.encodeList
        '''
        self.cap = cv2.VideoCapture(0)
        while True:
            ret, frame = self.cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            '''
            self.faceFrame = face_recognition.face_locations(frame)
            self.encodeFrame = face_recognition.face_encodings(
                frame, self.faceFrame)
            for encodeFace, faceLoc in zip(self.encodeFrame, self.faceFrame):
                self.matches = face_recognition.compare_faces(
                    self.encodeListKnown, encodeFace)
                self.faceDis = face_recognition.face_distance(
                    self.encodeListKnown, encodeFace)
                matchIndex = np.argmin(self.faceDis)
                if self.matches[matchIndex]:
                    name = className[matchIndex].upper()
                    y1, x2, y2, x1 = faceLoc
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, 'Unlocking', (250, 250),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                    
                    ########################################################
                    #
                    #time.sleep(2)                  
                    #pyautogui.click(x=845, y=622,button='left') 
                    ##time.sleep(2)
                    #pyautogui.write('123123')
                    #
                    ########################################################
                else:
                    y1, x2, y2, x1 = faceLoc
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, 'unauthorized', (240, 240),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
                        '''
            self.photo = PIL.ImageTk.PhotoImage(PIL.Image.fromarray(frame))
            self.canvas.create_image(-170, -170,
                                     image=self.photo, anchor=tk.NW)
            self.root.update()


Window()
