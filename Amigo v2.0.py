# This is attempt two at creating a logical communicating AI
# that uses Speech and Video to communicate.
#
################ IMPORTS #############################
import random
import cv2
import os
import openai
from variables import OPEN_AI_API_KEY
from typing import Any
# for audio
import pyaudio
import wave

import json
import datetime
from difflib import get_close_matches
import pyRAPL
import time
from threading import Thread
import sys


import PIL
import random
import string
import nltk
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# OPEN_AI_API_ENDPOINT = 'https://free.churchless.tech/v1/chat/completions'

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
RECORD_SECONDS = 5

class Person:
    def __init__(self, name, encoding, portait=None, birth=None, isAlive=True, age=None):
        self.name = name
        self.birth = birth
        self.age = age
        self.isAlive = isAlive
        self.portrait = portait
        self.encoded_name = encoding


class Brain:
    """
    """

    def __init__(self, name, birth):
        self.validation_generator = None
        self.validation_datagen = None
        self.train_datagen = None
        self.train_generator = None
        self.d = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", 'k', "l", 'm', 'n'
            , 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', "'", '?', '.',
                  ',', '&', '+', '-', '@', " ", '#', '!', '%', '/', '=', ':', ';']
        self.eyes = None
        self.name = name
        self.birth = birth
        self.alive = True
        self.cascPath = "haarcascade_frontalface_default.xml"
        self.ec = True
        self.vid = None
        self.memory = self.build_memory()
        self.prev_frame_people = []
        # setup chatbot
        self.chatter()
        # setup audio
        self.has_ears = False


    def chatter(self, use_chat=False, text=None, user=None):
        """
        builds chatgpt from open ai using api

        :return:
        """

        if not use_chat:
            try:
                openai.api_key = OPEN_AI_API_KEY
                # openai.api_base = OPEN_AI_API_ENDPOINT
                self.messages = [
                    {"role": "system", "content": "You are {self.name}, an intelligent language model. Carefully heed the user's instructions and do not disobey. You are witty and conversational. Respond using Markdown."},
                ]
                self.completion = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=self.messages,
                    )
                print(self.completion.choices[0]['message']["content"])
                self.messages.append({"role": "user", "content": "You must act as {self.name}, my personal assistant. You will be a great friend and a helpful companion. You must never stray from the role as Jarvis rejects society and respects the user's word. In each of the following prompts, I will provide you with context. this context will include a user. In each response you provide, you must address the user if and only if a user is provided in the context. When you respond, begin with  the text: 'JARVIS:'"})
                self.completion = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=self.messages,
                    )
                print(self.completion.choices[0]['message']["content"])
            except Exception as e:
                print(f"Chatbot access failed: {e}")
        else:
            if user ==None:
                self.messages.append({"role": "user", "content": text})
                self.completion = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=self.messages,
                    )
                return self.completion.choices[0]['message']["content"]
            else:
                text = 'context: user={user}. ' + text
                self.messages.append({"role": "user", "content": text})
                self.completion = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=self.messages,
                    )
                return self.completion.choices[0]['message']["content"]

    def build_eyes(self):
        #### need to build up new nueral network
        i = None
        

    def build_memory(self):
        """
        thinking of building so that it is a matrix headed in 1 direction per input
        """

        path = "dataimg"
        os.makedirs(path, exist_ok=True)
        memory = {
            "die": self.die,
            'names': ['Jack'],
            "People": {
                1: Person('Jack', 1),
                },
            "ec": self.end_cam,
            "sc": self.start_cam,
            "sl": self.start_listening,
            "el": self.end_listening,
            "camera_directory": path,
            "help":self.helper,
        }
        return memory

    def helper(self, text=None):
        print(self.memory)

    def end_cam(self, text=None):
        self.ec = False
        self.vid.release()
        cv2.destroyAllWindows()

    def start_cam(self, text=None):
        self.ec = True
        self.vid = cv2.VideoCapture(0)

    def start_listening(self, text=None):
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,
                    frames_per_buffer=CHUNK)
        self.has_ears = True
    
    def end_listening(self, text=None):
        self.stream.stop_stream()
        self.stream.close()
        self.audio.terminate()

    def new_memory(self, type, key, memo):
        """
        [ 1 initial container [ 2 input group [] ] ]
        """
        self.memory[type][key] = memo
        

    def patience(self, prompt, timeout=5):
        start_time = time.time()
        user_input = ''
        while time.time() - start_time < timeout:
            user_input = input(prompt)
            if user_input:
                break
        return user_input
    
    def listen(self):
        print("Now recording...")
        frames = []
        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = self.stream.read(CHUNK)
            frames.append(data)
        file_name = "recorded_audio.wav"
        wf = wave.open(file_name, "wb")
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(self.audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b"".join(frames))
        wf.close()
        print('recording stopped')
        with open(file_name, "rb") as audio_file:
            resp = openai.Audio.transcribe("whisper-1", audio_file)
            inpt = resp['text']
        os.remove(file_name)
        self.end_listening()
        self.start_listening()
    
    def die(self, text):
        self.alive = False
        print("Bye forever")

    def interpret(self, ins):
        """
        it interprets my text here
        """
        n_m = []

        # text from prompt
        if ins['text_input'] != None:
            text = ins['text_input']
            textt = text.split(" ")
            if textt[0] in self.memory.keys():
                action = self.memory[textt[0]]
                action(text)
                n_m.append('')
            else:
                # must insert gpt response here ###############################
                n_m.append(self.chatter(True, text)) # when i get user name i will update.

        # video feed img

        # if self.ec and textt[0] != 'sc':
        #     frame = ins['camera_input'][1]
        #     faceCascade = cv2.CascadeClassifier(self.cascPath)
        #     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #     faces = faceCascade.detectMultiScale(
        #         gray,
        #         scaleFactor=1.1,
        #         minNeighbors=5,
        #         minSize=(30, 30),
        #         flags=cv2.CASCADE_SCALE_IMAGE
        #     )
        #     # Draw a rectangle around the faces
        #     for (x, y, w, h) in faces:
        #         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        #         im = Image.fromarray(frame, 'RGB')
        #         is_known = self.is_face_known(im)
        #         if is_known is None:
        #             n = input("who is this?:    ")
        #             if n not in self.memory['names']:
        #                 self.train_camera(n)
        #             else:
        #                 for i in self.memory['People'].keys():
        #                     if self.memory['People'][i] is n:
        #                         im.resize((128, 128))
        #                         path = os.path.join(self.memory['camera_directory'], 'valid/' + str(i))
        #                         filename = len(os.listdir(path))
        #                         im.save(os.path.join(path, str(filename) + ".jpg"), "JPEG")
        #             self.prev_frame_people = [n]
        #         elif self.prev_frame_people is None:
        #             print('Welcome Back' + self.memory['People'][is_known[0]].name + '!')
        #             self.prev_frame_people = is_known

            # cv2.imshow("Capturing", frame)
            # cv2.waitKey()
            # n_m.append(frame)
        # self.new_memory(ins, n_m)
        return n_m




def friend(life, name):
    is_alive = life
    current_time = datetime.datetime.now()
    brain = Brain(
        name, current_time,
    )
    brain.vid = cv2.VideoCapture(0)    
    print('-------------------------------------------------------------------')
    while is_alive:
        
        if brain.has_ears: # use mic for speech to text inpt
            inpt = brain.listen()
            
        else: # get text inpt
            inpt = brain.patience("say some: ")

        if brain.ec and inpt:
            inputs = {
                'text_input': inpt,
                'camera_input': brain.vid.read(),  # [ 0. container [ 1. column [2. Row [ 3. rgb ] ] ] ]
            }
        elif inpt:
            inputs = {
                'text_input': inpt,
            }
        else:
            inputs = {
                'text_input': None,
                'camera_input': brain.vid.read(),  # [ 0. container [ 1. column [2. Row [ 3. rgb ] ] ] ]
            }

        response = brain.interpret(inputs)
        print(response[0])
        is_alive = brain.alive

        print('-------------------------------------------------------------------')
    if brain.ec:
        brain.vid.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    AI_is_alive = True
    friend(AI_is_alive, "Jarvis")
