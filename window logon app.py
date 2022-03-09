import tkinter as tk
import cv2
import PIL.Image
import PIL.ImageTk
import face_recognition as fr
import numpy as np
import pyautogui
import time


class Window:
    def __init__(self):
        self.root = tk.Tk()
        self.c_height = 350
        self.c_width = 300
        self.root.title('Hello')
        self.root.overrideredirect(True)
        self.root.geometry(f"{self.c_width}x{self.c_height}+{760}+{10}")
        self.root.resizable(False, False)
        self.root.configure(bg='light blue')
        self.T = tk.Label(self.root, text='RECOGNIZING',
                          font=('times', 15, 'bold'), bg='black', fg='white')
        self.T.pack(fill=tk.BOTH, expand=1)

        self.T1 = tk.Button(self.root, text='Quit',
                            font=('times', 15, 'bold'), bg='black', fg='white', command=self.root.destroy)
        self.T1.pack(fill=tk.BOTH, expand=1, side=tk.BOTTOM)
        self.canvas = tk.Canvas(
            self.root, width=self.c_width, height=self.c_height, bg='gray')
        self.canvas.pack(padx=5, pady=5)

        self.video()
        self.root.mainloop()

    def click(self):
        #print('text')
        #pyautogui.Click(x=422, y=496)
        #auto unlock screen
        #time.sleep(5)
        #pyautogui.press("space")
        time.sleep(1)
        pyautogui.click(x=845, y=622,button='left')
        time.sleep(1)
        pyautogui.write('123123')
        #print(self.text.config())

    def video(self):
        self.praveen_image = fr.load_image_file("praveen.jpg")
        praveen_face_encoding = fr.face_encodings(self.praveen_image)[0]
        known_face_encondings = [praveen_face_encoding]
        known_face_names = ["praveen"]
        self.cap = cv2.VideoCapture(0)
        count = 0
        while True:
            ret, frame = self.cap.read()
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = fr.face_locations(rgb_frame)
            face_encodings = fr.face_encodings(rgb_frame, face_locations)
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                matches = fr.compare_faces(
                    known_face_encondings, face_encoding)
                face_distances = fr.face_distance(
                    known_face_encondings, face_encoding)
                best_match_index = np.argmin(face_distances)
                name = "Unknown"
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

                    cv2.rectangle(frame, (left, top),
                                  (right, bottom), (255, 0, 0), 2)

                    cv2.rectangle(frame, (left, bottom - 35),
                                  (right, bottom), (255, 0, 0), cv2.FILLED)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(frame, name, (left + 6, bottom - 6),
                                font, 1.0, (0, 255, 0), 1)
                    if count == 0:
                        self.click()
                        quit()
                        count = count+1

            self.photo = PIL.ImageTk.PhotoImage(
                PIL.Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)))
            self.canvas.create_image(-170, -170,
                                     image=self.photo, anchor=tk.NW)
            self.root.update()


Window()
