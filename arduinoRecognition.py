import face_recognition
import cv2
import serial

# Initialize serial connection to Arduino
ser = serial.Serial('/dev/ttyACM0', 9600)

# Load a sample picture and learn how to recognize it.
known_image = face_recognition.load_image_file("known_person.jpg")
known_face_encoding = face_recognition.face_encodings(known_image)[0]

# Initialize some variables
face_locations = []
face_encodings = []

# Start capturing video from the first camera device
video_capture = cv2.VideoCapture(0)

while True:
    # Capture a single frame from the video source
    ret, frame = video_capture.read()

    # Find all face locations and face encodings in the frame
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    # Check each face in the current frame against the known face encoding
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces([known_face_encoding], face_encoding)
        
        if True in matches:
            # If the face matches the known face, draw a box around it
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            ser.write(b'1')  # Send '1' to Arduino when a known face is detected
        else:
            ser.write(b'0')  # Send '0' otherwise

    # Display the frame with boxes around recognized faces
    cv2.imshow('Video', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture
video_capture.release()
cv2.destroyAllWindows()
ser.close()
