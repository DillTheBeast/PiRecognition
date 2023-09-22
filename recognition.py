import cv2
import face_recognition

# Load the pictures of the family so I can compare
img = cv2.imread("Dillon.jpg")
rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_encoding = face_recognition.face_encodings(rgb_img)[0]
img2 = cv2.imread("Arielle.jpg")
rgb_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
img_encoding2 = face_recognition.face_encodings(rgb_img2)[0]
img3 = cv2.imread("Terry.jpg")
rgb_img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
img_encoding3 = face_recognition.face_encodings(rgb_img3)[0]
# img4 = cv2.imread("Terrielle.jpg")
# rgb_img4 = cv2.cvtColor(img4, cv2.COLOR_BGR2RGB)
# img_encoding4 = face_recognition.face_encodings(rgb_img4)[0]

#Setting up video with lower res so les lag
video_capture = cv2.VideoCapture(0)
video_capture.set(3, 640)  # Set width
video_capture.set(4, 480)  # Set height

while True:
    result, video_frame = video_capture.read()
    if result is False:
        break

    # Convert the video frame to RGB for face_recognition
    rgb_video_frame = cv2.cvtColor(video_frame, cv2.COLOR_BGR2RGB)
    
    # Using face_recognition's face detection
    face_locations = face_recognition.face_locations(rgb_video_frame)

    frame_encodings = face_recognition.face_encodings(rgb_video_frame, face_locations)

    #Going through faces
    for face_location, face_encoding in zip(face_locations, frame_encodings):
        top, right, bottom, left = face_location
        #Making variable for comparing faces
        matches = face_recognition.compare_faces([img_encoding], face_encoding, tolerance=0.5)
        matches2 = face_recognition.compare_faces([img_encoding2], face_encoding, tolerance=0.5)
        matches3 = face_recognition.compare_faces([img_encoding3], face_encoding, tolerance=0.5)
        #matches4 = face_recognition.compare_faces([img_encoding4], face_encoding, tolerance=0.5)
        #Finding out if they match
        if matches[0]:
            #Drawing a box with correct name around face
            cv2.rectangle(video_frame, (left, top), (right, bottom), (243, 255, 130), 4)
            cv2.putText(video_frame, "Dillon", (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        elif matches2[0]:
            cv2.rectangle(video_frame, (left, top), (right, bottom), (180, 105, 255), 4)
            cv2.putText(video_frame, "Arielle", (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        elif matches3[0]:
            cv2.rectangle(video_frame, (left, top), (right, bottom), (0, 255, 0), 4)
            cv2.putText(video_frame, "Terry", (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        # elif matches4[0]:
        #     cv2.rectangle(video_frame, (left, top), (right, bottom), (0, 0, 0), 4)
        #     cv2.putText(video_frame, "Terrielle", (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
        else: 
            #If face isn't recognized Drawing stranger box
            cv2.rectangle(video_frame, (left, top), (right, bottom), (255, 0, 0), 4)
            cv2.putText(video_frame, "Stranger", (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    #Setting up the actual displayed window
    cv2.imshow("Face Detection", video_frame)

    #If the user presses q key then break out of loop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

#Stop everything
video_capture.release()
cv2.destroyAllWindows()



#Works with picture
# import cv2
# import matplotlib.pyplot as plt

# imagePath = 'Dillon.jpg'
# img = cv2.imread(imagePath)

# gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# face_classifier = cv2.CascadeClassifier(
#     cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
# )
# face = face_classifier.detectMultiScale(
#     gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
# )

# for (x, y, w, h) in face:
#     cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 4)

# img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# plt.figure(figsize=(20,10))
# plt.imshow(img_rgb)
# plt.axis('off')
# plt.show()