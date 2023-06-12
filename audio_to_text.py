import speech_recognition as sr

r = sr.Recognizer()

def listen(text_label):
    with sr.Microphone() as source:
        audio_data = r.listen(source)
        try:
            text = r.recognize_google(audio_data)
            text_label.configure(text=text)
        except sr.UnknownValueError:
            pass
