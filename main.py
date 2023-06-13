'''
Code written by Pratik Kharel
For Final Project in Course: ICS4UI

Main Function of the code:
-Online Talk Assistant. 
-Has memory system to recognize known users. 
-Built-in hand detection that recognizes ASL to understand input from mute/deaf users. 
-Generates responses and does simple tasks for the user on their computer.
-Simple tasks include searching the internet, browsing websites, answering questions.
*Program is limited in terms of processing power and input limitations due to the useage of Open Ai's API.
This company has a free trial of their API which allows the code to work for a set amount of runs. 
Getting more inputs will require the use of real-world currency which, for me, is limited.
Future implementations will have no limit once a stable system for responses is added.*
'''

# Import OpenCV library for computer vision tasks
import cv2
# Import Tkinter library for creating graphical user interfaces
from tkinter import *
# Import Image and ImageTk classes from PIL library for working with images
from PIL import Image, ImageTk
# Import custom hand_detection function for detecting hands in images
from hand_detection import hand_detection
# Import custom listen function for transcribing audio input to text
from audio_to_text import listen

# Set up video capture using OpenCV
# 0 indicates that the default camera should be used
cap = cv2.VideoCapture(0)
# Set the width and height of the video capture
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# Get the width and height of the video capture
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Set up the main window using Tkinter
window = Tk()
# Set the title of the window
window.title("Live Video Input with Hand Detection, Facial Detection, and Audio to Text Conversion")
# Set the size of the window to be 1.25 times larger than the video capture dimensions
window.geometry(f"{int(width*1.25)}x{int(height*1.25)}")

# Set up a canvas within the window to display elements
canvas = Canvas(window, width=int(width*1.25), height=int(height*1.25))
canvas.pack()

# Set up a label within the canvas to display the video capture
label = Label(canvas)
label.place(relx=0.125, rely=0.1, relwidth=0.75, relheight=0.8)

# Set up a label within the canvas to display text from audio input
text_label = Label(canvas, font=("Helvetica", 16))
text_label.place(relx=0.125, rely=0, relwidth=0.75, relheight=0.1)

# Function to show a frame from the video capture on the label
def show_frame():
    # Read a frame from the video capture
    _, frame = cap.read()
    # Apply hand detection to the frame using a custom function
    frame = hand_detection(frame)
    # Flip the frame horizontally so that it appears as a mirror image
    frame = cv2.flip(frame, 1)
    # Convert the frame from a NumPy array to a PIL Image
    img = Image.fromarray(frame)
    # Convert the PIL Image to a Tkinter PhotoImage
    imgtk = ImageTk.PhotoImage(image=img)
    # Update the label with the new PhotoImage
    label.imgtk = imgtk
    label.configure(image=imgtk)
    # Call this function again after 10ms to update with a new frame
    label.after(10, show_frame)

# Function to listen for audio input and update text label with transcription
def on_listen():
    # Call custom listen function and pass in text label as argument
    listen(text_label)
    # Update window to show changes to text label
    window.update_idletasks()

# Set up a button within the canvas to trigger listening for audio input
button = Button(canvas, text="Listen", command=on_listen, font=("Helvetica", 16), bg="black", fg="white")
button.place(relx=0.375, rely=0.9, relwidth=0.25, relheight=0.1)

# Start showing frames from video capture on label
show_frame()
# Run main loop of window to keep it open and responsive to user input
window.mainloop()
