# Import OpenCV library for computer vision tasks
import cv2
# Import os library for interacting with the operating system
import os

# Create face cascade classifier using pre-trained Haar cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Define face_detection function to detect faces in a given frame
def face_detection(frame):
    # Convert frame from BGR to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect faces in grayscale frame using face cascade classifier
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    # Loop through each detected face
    for i, (x,y,w,h) in enumerate(faces):
        # Draw rectangle around face on frame
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        # Extract face from frame
        face = frame[y:y+h, x:x+w]
        # Save face to file
        cv2.imwrite(os.path.join("faces", f"face_{i}.jpg"), face)
    # Return modified frame with rectangles drawn around faces
    return frame
