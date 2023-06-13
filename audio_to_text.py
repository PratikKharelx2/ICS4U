# Import speech_recognition library for speech recognition tasks
import speech_recognition as sr
# Import custom func function for generating text responses
from response_generator import func
# Import pyttsx3 library for text-to-speech conversion
import pyttsx3

# Create recognizer object from speech_recognition library
r = sr.Recognizer()
# Create text-to-speech engine object from pyttsx3 library
engine = pyttsx3.init()

# Define listen function to listen for audio input and update text label with transcription
def listen(text_label):
    # Use microphone as audio source
    with sr.Microphone() as source:
        # Listen for audio input
        audio_data = r.listen(source)
        try:
            # Recognize speech in audio data using Google's speech recognition API
            text = r.recognize_google(audio_data)
            # Generate response to recognized text using custom func function
            response = func(text)
            # Update text label with generated response
            text_label.configure(text=response)
            # Use text-to-speech engine to speak generated response
            engine.say(response)
            engine.runAndWait()
        except sr.UnknownValueError:
            # If speech could not be recognized, do nothing
            pass
