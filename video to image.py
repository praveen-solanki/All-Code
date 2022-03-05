import cv2
import os
import pickle

cap = cv2.VideoCapture(0)
text = "Wait for 3 Seconds"
text2 = "look at the Camera"
path = 'C:\\Users\\praveen solanki\\Desktop\\face recignition\\images\\'

while True:
    ret, frame = cap.read()
    video = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    inputname = input("Enter your name : ")
    print("Are you ready for photo! ")
    answer = input("y or n : ")
    if answer == 'y' or answer == 'Y':
        print("Smile!")
        cv2.putText(video,text, (10,90),cv2.FONT_HERSHEY_SIMPLEX,2,(100,10,34),4)
        cv2.putText(video,text2, (10,150),cv2.FONT_HERSHEY_SIMPLEX,2,(100,10,34),5)
        cv2.imshow('video',video)
        cv2.waitKey(3000)
        if ret:
            name = path + inputname +'.jpg'
            print("Sucessfully captured")
            #file = open('image.pkl', 'wb') 
            #pickle.dump(frame, file)
            #file.close()
            cv2.imwrite(name,frame)
            break
    else:
        continue

    if cv2.waitKey(24) == 27:
        break
cap.release()
cv2.destroyAllWindows()