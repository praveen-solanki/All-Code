import face_recognition
import cv2
import pickle

# Load face encodings and names from the pickle file
with open('face_encodings.pkl', 'rb') as f:
    known_face_encodings, known_face_names = pickle.load(f)


v = 'C:\\Users\\praveen solanki\\Desktop\\AMS\\attendence\\codes\\3.mp4'
# Load video from file or capture from webcam
video_capture = cv2.VideoCapture(0)  # For video file
# video_capture = cv2.VideoCapture(0)  # For webcam

Roll_list= set()

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    if not ret:
        break

    # Resize the frame for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    
    # Convert the image from BGR color (OpenCV uses) to RGB (face_recognition uses)
    #rgb_small_frame = small_frame[:, :, ::-1]

    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Find all faces and face encodings in the current frame
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for face_encoding, face_location in zip(face_encodings, face_locations):
        # See if the face matches any known face encodings
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        new_name = "Unknown"

        # Use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = face_distances.argmin()
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

            Roll_No = name[:6]
            #print(Roll_No)
            new_name = name[6:]
            #print(new_name)

            Roll_list.add(Roll_No)

        # Draw a box around the face and label it with the name
        top, right, bottom, left = face_location
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)

        

        cv2.putText(frame, new_name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 1)

    # Display the resulting frame
    frame = cv2.resize(frame, (720, 480))
    cv2.imshow('Video', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


print("Roll NO.:", Roll_list)
# Release the capture and close windows
video_capture.release()
cv2.destroyAllWindows()
