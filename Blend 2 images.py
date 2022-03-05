import cv2
import numpy as np

img = cv2.cv2.imread('SEE.jfif')
img = cv2.resize(img, (700, 600))
img1 = cv2.imread('ICEMOUNTAIN.jfif')
img1 = cv2.resize(img1, (700, 600))


def blend(x):
    pass


img2 = np.zeros((499, 499, 3), np.uint8)
cv2.namedWindow("window")

cv2.createTrackbar("a", "window", 1, 100, blend)
switch = "0:OFF\n1:ON"
cv2.createTrackbar(switch, "window", 0, 1, blend)

while True:
    s = cv2.getTrackbarPos(switch, "window")
    a = cv2.getTrackbarPos("a", "window")
    n = float(a/100)

    if s == 0:
        result = img2[:]
    else:
        result = cv2.addWeighted(img, 1-n, img1, n, 0)
        cv2.putText(result, str(a), (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 2)
    cv2.imshow('result', result)
    if cv2.waitKey(1) == 27:
        break

cv2.waitKey(0)
cv2.destroyAllWindows()