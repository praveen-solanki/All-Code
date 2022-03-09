
import cv2
import dlib

img = cv2.imread('praveen.jpg')
img = cv2.resize(img, (0,0), None, 0.5, 0.5)

imgOriginal = img.copy()

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('models/face landmark using dlib/shape_predictor_68_face_landmarks.dat')

imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = detector(imgGray)

for face in faces:
    x1,y1 = face.left(), face.top()
    x2, y2 = face.right(), face.bottom()
    imgOriginal = cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
    landmarks = predictor(imgGray, face)
    for n in range(68):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        cv2.circle(imgOriginal, (x,y), 5, (50, 50, 255), cv2.FILLED)

cv2.imshow('dfd', imgOriginal)
cv2.waitKey()