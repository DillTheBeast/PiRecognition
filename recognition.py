import cv2
import time
import face_recognition

img = cv2.imread("Messi1.webp")
#Switching color type
rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_encoding = face_recognition.face_encodings(rgb_img)[0]

img2 = cv2.imread("Elon Musk.jpg")
#Switching color type
rgb_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
img_encoding2 = face_recognition.face_encodings(rgb_img2)[0]

cv2.imshow("IMG", img)
cv2.imshow("IMG", img2)
cv2.waitKey(0)

time.sleep(1)
cv2.destroyAllWindows()
quit()