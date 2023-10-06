# # Program to train with the faces and create a YAML file

# import cv2  # For Image processing
# import numpy as np  # For converting Images to Numerical array
# import os  # To handle directories
# from PIL import Image  # Pillow lib for handling images

# # Load the face cascade classifier
# face_cascade = cv2.CascadeClassifier('dillon_frontalface_default.xml')

# # Create the LBPH face recognizer
# #recognizer = cv2.face.createLBPHFaceRecognizer()
# # Create the LBPH face recognizer
# recognizer = cv2.face.LBPHFaceRecognizer_create()

# # Initialize variables
# Face_ID = -1
# pev_person_name = ""
# y_ID = []
# x_train = []

# # Specify the path to the directory containing face images
# Face_Images = os.path.join(os.getcwd(), "Face_Images")
# print(Face_Images)

# # Traverse the directory tree and process each image
# for root, dirs, files in os.walk(Face_Images):
#     for file in files:
#         # Check if the file is an image (ends with jpeg, jpg, or png)
#         if file.endswith("jpeg") or file.endswith("jpg") or file.endswith("png"):
#             path = os.path.join(root, file)
#             person_name = os.path.basename(root)
#             print(path, person_name)

#             # Check if the name of the person has changed
#             if pev_person_name != person_name:
#                 Face_ID = Face_ID + 1  # Increment the ID count
#                 pev_person_name = person_name

#             # Open the image and convert it to grayscale
#             Gery_Image = Image.open(path).convert("L")

#             # Resize the grayscale image to 800x800 pixels
#             #Crop_Image = Gery_Image.resize((800, 800), Image.ANTIALIAS)
#             Crop_Image = Gery_Image.resize((800, 800), Image.BICUBIC)

#             # Convert the resized image to a numpy array
#             Final_Image = np.array(Crop_Image, "uint8")

#             # Detect faces in the image
#             faces = face_cascade.detectMultiScale(Final_Image, scaleFactor=1.5, minNeighbors=5)
#             print(Face_ID, faces)

#             # Crop the region of interest (ROI) for each detected face
#             for (x, y, w, h) in faces:
#                 roi = Final_Image[y:y + h, x:x + w]
#                 x_train.append(roi)
#                 y_ID.append(Face_ID)

# # Train the recognizer with the collected face samples
# recognizer.train(x_train, np.array(y_ID))

# # Save the trained recognizer as a YAML file
# recognizer.save("face-trainner.yml")

# Program to train with the faces and create a YAML file

import cv2  # For Image processing
import numpy as np  # For converting Images to Numerical array
import os  # To handle directories
from PIL import Image  # Pillow lib for handling images

# Load the face cascade classifier
face_cascade = cv2.CascadeClassifier('dillon_frontalface_default.xml')

# Create the LBPH face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Initialize variables
Face_ID = -1
pev_person_name = ""
y_ID = []
x_train = []

# Specify the path to the directory containing face images
Face_Images = os.path.join(os.getcwd(), "Face_Images")
print("Directory for Face_Images:", Face_Images)

# Check if the Face_Images directory exists
if not os.path.exists(Face_Images):
    print("Face_Images directory does not exist!")
    exit()

# Traverse the directory tree and process each image
for root, dirs, files in os.walk(Face_Images):
    # Debug: Print the directory being processed
    print("Processing directory:", root)
    
    for file in files:
        # Check if the file is an image (ends with jpeg, jpg, or png)
        if file.endswith("jpeg") or file.endswith("jpg") or file.endswith("png"):
            path = os.path.join(root, file)
            person_name = os.path.basename(root)
            print("Processing image:", path, "| Person:", person_name)

            # Check if the name of the person has changed
            if pev_person_name != person_name:
                Face_ID = Face_ID + 1  # Increment the ID count
                pev_person_name = person_name

            # Open the image and convert it to grayscale
            Gery_Image = Image.open(path).convert("L")

            # Resize the grayscale image to 800x800 pixels
            Crop_Image = Gery_Image.resize((800, 800), Image.BICUBIC)

            # Convert the resized image to a numpy array
            Final_Image = np.array(Crop_Image, "uint8")

            # Detect faces in the image
            faces = face_cascade.detectMultiScale(Final_Image, scaleFactor=1.5, minNeighbors=5)
            print("Face ID:", Face_ID, "| Detected faces:", faces)

            # Crop the region of interest (ROI) for each detected face
            for (x, y, w, h) in faces:
                roi = Final_Image[y:y + h, x:x + w]
                x_train.append(roi)
                y_ID.append(Face_ID)

# Train the recognizer with the collected face samples
recognizer.train(x_train, np.array(y_ID))

# Save the trained recognizer as a YAML file
recognizer.save("face-trainner.yml")
