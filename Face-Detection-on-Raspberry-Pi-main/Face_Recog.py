import cv2
import numpy as np
import os
from PIL import Image

labels = ["Dillon", "Terrielle", "Arielle", "Terry"]  # List of labels for face recognition

# Load the pre-trained face cascade classifier
face_cascade = cv2.CascadeClassifier('dillon_frontalface_default.xml')
print(cv2.__version__)

# Load the pre-trained face recognizer
#recognizer = cv2.face.createLBPHFaceRecognizer()
recognizer = cv2.face.LBPHFaceRecognizer_create()
#recognizer.load("face-trainner.yml")
recognizer.read("face-trainner.yml")

# Open the video capture device (webcam)
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the video capture device
    ret, img = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)

    # Iterate over each detected face
    for (x, y, w, h) in faces:
        # Extract the region of interest (ROI) from the grayscale frame
        roi_gray = gray[y:y+h, x:x+w]

        # Perform face recognition on the ROI
        id_, conf = recognizer.predict(roi_gray)

        # If the confidence is above a certain threshold, display the recognized label on the frame
        if conf >= 80:
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            cv2.putText(img, name, (x, y), font, 1, (0, 0, 255), 2)

        # Draw a rectangle around the detected face on the frame
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the frame with the detected faces
    cv2.imshow('Preview', img)

    # Check if the 'q' key is pressed to exit the loop
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# Release the video capture device and destroy all windows
cap.release()
cv2.destroyAllWindows()