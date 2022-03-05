import cv2
import numpy as np
import pyautogui as pg

# Create Resolution
rs = pg.size()

# File name in which we store recording
f = "outputVideo.avi"
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video = cv2.VideoWriter(f, fourcc, 10, rs)

# Creating recording module
cv2.namedWindow("LiveRecording", cv2.WINDOW_NORMAL)
cv2.resizeWindow("LiveRecording", (720, 480))
print("Recording")
while True:
    
    frame = pg.screenshot()
    image = np.array(frame)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    video.write(image)
    #cv2.imshow("video", image)
    if cv2.waitKey(1) == 27:
        break
print("Recording Completed")
video.release()
cv2.destroyAllWindows()
