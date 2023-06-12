import cv2
import os

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def face_detection(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for i, (x,y,w,h) in enumerate(faces):
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        face = frame[y:y+h, x:x+w]
        cv2.imwrite(os.path.join("faces", f"face_{i}.jpg"), face)
    return frame
