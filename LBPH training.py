import os
import cv2
import numpy as np
from PIL import Image

names = []
path = []

# Get the names of all the users
for users in os.listdir("myimage/testing"):
    names.append(users)

# Get the path to all the images
for name in names:
    for image in os.listdir("myimage/testing/{}".format(name)):
        path_string = os.path.join("myimage/testing/{}".format(name), image)
        path.append(path_string)

faces = []
ids = []

# For each image create a numpy array and add it to faces list
for img_path in path:
    image = Image.open(img_path).convert("L")
    imgNp = np.array(image, "uint8")
    #id = int(img_path.split("/")[2].split("_")[0])
    id = 0
    faces.append(imgNp)
    ids.append(id)

# Convert the ids to numpy array and add it to ids list
ids = np.array(ids)

# Call the recognizer
trainer = cv2.face.LBPHFaceRecognizer_create()
# Give the faces and ids numpy arrays
trainer.train(faces, ids)
# Write the generated model to a yml file
trainer.write("training.yml")

print("[INFO] Training Done")