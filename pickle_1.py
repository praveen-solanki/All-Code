import face_recognition
import cv2
import os
import pickle

# Folder containing images to encode
image_folder = 'attendence/image/'

# List to hold encodings and names
known_face_encodings = []
known_face_names = []

# Loop through each image in the folder
for filename in os.listdir(image_folder):
    if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
        # Load the image
        image_path = os.path.join(image_folder, filename)
        image = face_recognition.load_image_file(image_path)

        # Get face encodings (assumes one face per image)
        face_encodings = face_recognition.face_encodings(image)

        if face_encodings:
            # Use the filename (without extension) as the name
            name = os.path.splitext(filename)[0]

            # Store the encoding and the name
            known_face_encodings.append(face_encodings[0])
            known_face_names.append(name)

# Save the encodings and names to a pickle file
with open('face_encodings.pkl', 'wb') as f:
    pickle.dump((known_face_encodings, known_face_names), f)

print("Face encodings saved successfully!")
