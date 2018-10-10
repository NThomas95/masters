import naoqi
from naoqi import ALProxy


tts = ALProxy("ALTextToSpeech", "192.168.1.5", 9559)
tts.say("Welcome to the dynamic systems and control laboratory at San Diego State University")
