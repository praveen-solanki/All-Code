import cv2
import numpy as np


cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    
    #img = cv2.imread("praveen.jpg")

    # 1) Edges
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)

    # 2) Color
    color = cv2.bilateralFilter(frame, 9, 300, 300)

    # 3) Cartoon
    cartoon = cv2.bitwise_and(color, color, mask=edges)


    #cv2.imshow("Image", img)
    cv2.imshow("Cartoon", cartoon)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
