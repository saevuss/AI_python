#We are going to use Google Speech API in Python to make recognition happen
import pyaudio
import speech_recognition as sr
#creating an object
recording = sr.Recognizer()
with sr.Microphone(sample_rate=16000) as source:
    recording.adjust_for_ambient_noise(source)
    print("Please say something...")
    audio = recording.listen(source)

    #now google API would recognize the voice and gives the output
    try:
        print('You said: \n' + recording.recognize_google(audio))
    except Exception as e:
        print(e)








