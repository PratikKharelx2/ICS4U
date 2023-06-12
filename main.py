import cv2
from tkinter import *
from PIL import Image, ImageTk
from hand_detection import hand_detection
from face_detection import face_detection
from audio_to_text import listen

cap = cv2.VideoCapture(0)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

window = Tk()
window.title("Live Video Input with Hand Detection, Facial Detection, and Audio to Text Conversion")
window.geometry(f"{int(width*1.25)}x{int(height*1.25)}")

canvas = Canvas(window, width=int(width*1.25), height=int(height*1.25))
canvas.pack()

label = Label(canvas)
label.place(relx=0.125, rely=0.1, relwidth=0.75, relheight=0.8)

text_label = Label(canvas, font=("Helvetica", 16))
text_label.place(relx=0.125, rely=0, relwidth=0.75, relheight=0.1)

is_listening = BooleanVar()
is_listening.set(False)

def show_frame():
    _, frame = cap.read()
    frame = face_detection(frame)
    frame = hand_detection(frame)
    if is_listening.get():
        frame = cv2.blur(frame, (30, 30))
    frame = cv2.flip(frame, 1)
    img = Image.fromarray(frame)
    imgtk = ImageTk.PhotoImage(image=img)
    label.imgtk = imgtk
    label.configure(image=imgtk)
    label.after(10, show_frame)

def on_listen():
    is_listening.set(True)
    listen(text_label)
    is_listening.set(False)

button = Button(canvas, text="Listen", command=on_listen, font=("Helvetica", 16), bg="black", fg="white")
button.place(relx=0.375, rely=0.9, relwidth=0.25, relheight=0.1)

show_frame()
window.mainloop()
