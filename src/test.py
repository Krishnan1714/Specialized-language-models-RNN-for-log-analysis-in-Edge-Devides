import speech_recognition as sr

recognizer = sr.Recognizer()
with sr.Microphone() as source:
    print("Say something...")
    recognizer.adjust_for_ambient_noise(source, duration=1)
    audio = recognizer.listen(source)

try:
    print("You said: " + recognizer.recognize_google(audio))
except Exception as e:
    print("Error:", e)
