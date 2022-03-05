
import cv2
import numpy as np
img = cv2.imread('LION.jfif')

# For making border images
# ( image, thickness, thickness, thickness, thickness,Border type,color )
b = cv2.copyMakeBorder(img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[0,0,0])

cv2.imshow("image", b)
cv2.waitKey(0)
cv2.destroyAllWindows()

