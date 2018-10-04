import naoqi
from naoqi import ALProxy


tts = ALProxy("ALTextToSpeech", "192.168.1.5", 9559)
tts.say("Hello Dr. Mousavi. I am NAO. How are you?")
