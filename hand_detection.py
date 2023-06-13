# Import OpenCV library for computer vision tasks
import cv2
# Import mediapipe library for machine learning solutions
import mediapipe as mp

# Create hands object from mediapipe solutions
mpHands = mp.solutions.hands
hands = mpHands.Hands()
# Create drawing utilities object from mediapipe solutions
mpDraw = mp.solutions.drawing_utils

# Define hand_detection function to detect hands in a given frame
def hand_detection(frame):
    # Convert frame from BGR to RGB color space
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Process frame using hands object to detect hands
    results = hands.process(frame)
    # Check if any hand landmarks were detected
    if results.multi_hand_landmarks:
        # Loop through each detected hand
        for handLms in results.multi_hand_landmarks:
            # Draw hand landmarks and connections on frame
            mpDraw.draw_landmarks(frame, handLms, mpHands.HAND_CONNECTIONS)
    # Return modified frame with hand landmarks drawn on it
    return frame
