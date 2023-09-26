# this is a script for an AI that won't die Whose program doesn't end.
# only constantly consumes the energy of my laptop until it is too great to be feed here
# THis will be an AI attempt to communicate by img and text response.


########################################################################################################################
#####################                                    NEEDS
########################################################################################################################
#####################  1. determine how to get similarity score. store each memory by its score.
#####################  2. y by its s
#####################  3. get audio
#####################  4. audio to text
#####################  5. setup local host for output img
#####################  6. create awareness lol
#####################  7. display self on the local host
#####################  8. add captions and allow constant refresh after activity and time to decide if there is an input
#####################  9. setup to run on quantum cloud qiskit
#####################  10. need to process all inputs separately and store them separately
#####################  NLP,  img2img, text2img, response to caption, method to compare two faces!
#####################  Painting for background. mode that starts small and updates each time
#####################  Locking mode if unkown person does not have password


##### USING ######
from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer
import random
import cv2
import os
from PIL import Image
from keras import models
from keras import layers
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
##################

from typing import Any

import json
import datetime
from difflib import get_close_matches
import pyRAPL
import time
from threading import Thread
import sys

import string
import nltk
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer  # It has the ability to lemmatize.
import tensorflow as tf  # A multidimensional array of elements is represented by this symbol.
from tensorflow.keras import Sequential  # Sequential groups a linear stack of layers into a tf.keras.Model
from tensorflow.keras.layers import Dense, Dropout


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
        brain design:
            uses 4-D wave function
            maps: by classification category
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
        self.chatter = ChatBot(name, preprocessors=['chatterbot.preprocessors.clean_whitespace',
                                                    'chatterbot.preprocessors.unescape_html',
                                                    'chatterbot.preprocessors.convert_to_ascii'])
        self.trainer = ChatterBotCorpusTrainer(self.chatter)
        self.trainer.train("chatterbot.corpus.english")
        self.trainer.train("chatterbot.corpus.english.greetings")
        self.trainer.train("chatterbot.corpus.english.conversations")
        self.trainer.train("chatterbot.corpus.english.conversations")
        self.classification_number = 2
        self.build_eyes()
        self.age = datetime.datetime.now() - self.birth
        # self.consumed = meter.result

    def build_eyes(self):
        size = 128
        self.train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)
        self.validation_datagen = ImageDataGenerator(rescale=1.255)
        self.train_generator = self.train_datagen.flow_from_directory('dataimg/training',
                                                                      target_size=(size, size), batch_size=64,
                                                                      class_mode='binary')
        self.validation_generator = self.validation_datagen.flow_from_directory('dataimg/valid',
                                                                                target_size=(size, size),
                                                                                batch_size=64, class_mode='binary')
        try:
            self.eyes = models.load_model('eyes.h5')
            self.eyes.fit_generator(self.train_generator, epochs=4, steps_per_epoch=63,
                                    validation_data=self.validation_generator, validation_steps=7, workers=4)
        except:
            self.eyes = models.Sequential()
            self.eyes.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(size, size, 3)))
            self.eyes.add(layers.MaxPooling2D((2, 2)))
            self.eyes.add(layers.Conv2D(64, (3, 3), activation='relu'))
            self.eyes.add(layers.MaxPooling2D(2, 2))
            self.eyes.add(layers.Conv2D(128, (3, 3), activation='relu'))
            self.eyes.add(layers.MaxPooling2D(2, 2))
            self.eyes.add(layers.Conv2D(128, (3, 3), activation='relu'))
            self.eyes.add(layers.MaxPooling2D(2, 2))
            self.eyes.add(layers.Flatten())
            self.eyes.add(layers.Dropout(0.5))
            self.eyes.add(layers.Dense(512, activation='relu'))
            self.eyes.add(layers.Dense(1, activation='sigmoid'))
            self.eyes.compile(optimizer=optimizers.RMSprop(lr=0.0003), loss='binary_crossentropy', metrics=['acc'])
            #
            # self.train_datagen = ImageDataGenerator(
            #     rescale=1. / 255,
            #     rotation_range=40,
            #     width_shift_range=0.2,
            #     height_shift_range=0.2,
            #     shear_range=0.2,
            #     zoom_range=0.2,
            #     horizontal_flip=True)
            # self.validation_datagen = ImageDataGenerator(rescale=1.255)
            # self.train_generator = train_datagen.flow_from_directory('dataimg/training',
            #                                                          target_size=(size, size), batch_size=64,
            #                                                          class_mode='binary')
            # self.validation_generator = validation_datagen.flow_from_directory('dataimg/valid',
            #                                                                    target_size=(size, size),
            #                                                                    batch_size=64, class_mode='binary')
            #
            self.eyes.fit_generator(train_generator, epochs=5, steps_per_epoch=63,
                                    validation_data=validation_generator, validation_steps=7, workers=4)
            self.eyes.save('eyes.h5')

    def build_memory(self):
        """
        thinking of building so that it is a matrix headed in 1 direction per input
        """

        path = "dataimg"
        os.makedirs(path, exist_ok=True)
        memory = {
            'd': self.dictionary,
            'dice': self.die_roll,
            "die": self.die,
            'names': ['Jack'],
            "People": {
                1: Person('Jack', 1),
            },
            "ec": self.end_cam,
            "sc": self.start_cam,
            "camera_directory": path,
            "help":self.helper,
        }
        return memory

    def helper(self, text):
        print(self.memory)

    def end_cam(self, text):
        self.ec = False
        self.vid.release()
        cv2.destroyAllWindows()

    def start_cam(self, text):
        self.ec = True
        self.vid = cv2.VideoCapture(0)

    def new_memory(self, type, key, memo):
        """
        [ 1 initial container [ 2 input group [] ] ]
        """
        self.memory[type][key] = memo

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
                n_m.append(self.chatter.get_response(text))
        # n_m.append(self.text_to_matrix(text)) #deprecated

        # video feed img

        if self.ec and textt[0] != 'sc':
            frame = ins['camera_input'][1]
            faceCascade = cv2.CascadeClassifier(self.cascPath)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            # Draw a rectangle around the faces
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                im = Image.fromarray(frame, 'RGB')
                is_known = self.is_face_known(im)
                if is_known is None:
                    n = input("who is this?:    ")
                    if n not in self.memory['names']:
                        self.train_camera(n)
                    else:
                        for i in self.memory['People'].keys():
                            if self.memory['People'][i] is n:
                                im.resize((128, 128))
                                path = os.path.join(self.memory['camera_directory'], 'valid/' + str(i))
                                filename = len(os.listdir(path))
                                im.save(os.path.join(path, str(filename) + ".jpg"), "JPEG")
                    self.prev_frame_people = [n]
                elif self.prev_frame_people is None:
                    print('Welcome Back' + self.memory['People'][is_known[0]].name + '!')
                    self.prev_frame_people = is_known

            # cv2.imshow("Capturing", frame)
            # cv2.waitKey()
            n_m.append(frame)

        # audio input
        # audio = ins['audio_input']

        # self.new_memory(ins, n_m)
        return n_m

    def die(self, words):
        self.alive = False
        print("Bye")

    def learn(self):
        """
        This is an activity to perform by itself in which it pulls in data from google
        """

    def social(self):
        """
        I connect it to facebook to interact with others...
        """

    def text_to_matrix(self, inp):
        tm = []
        if inp is None:
            return []
        for char in inp:
            if char in self.d:
                tm.append(1)
            else:
                tm.append(0)
        return tm

    def dictionary(self, t):
        data = json.load(open("dictionary.json"))
        words = t.split(' ')
        words.remove('d')
        for word in words:
            w = word.lower()

            if w in data:
                output = data[w]
            # for getting close matches of word
            elif len(get_close_matches(w, data.keys())) > 0:
                yn = input(
                    "Did you mean % s instead? Enter Y if yes, or N if no: " % get_close_matches(w, data.keys())[0])
                yn = yn.lower()
                if yn == "y":
                    output = data[get_close_matches(w, data.keys())[0]]
                elif yn == "n":
                    output = "The word doesn't exist. Please double check it."
                else:
                    output = "We didn't understand your entry."
            else:
                output = "The word doesn't exist. Please double check it."

            if type(output) == list:
                for item in output:
                    print(item)
            else:
                print(output)

    def die_roll(self, t):
        def roll(number, odds=1000):
            the_flop = []
            for i in range(odds):
                the_flop.append(random.randint(1, number))
            return round(sum(the_flop)/odds)
        want_to_die = True
        words = t.split()
        num = words[1]
        bad = True
        while bad:
            try:
                num = int(num)
                bad = False
            except:
                num = input("Enter a number idiot: ")
        choices = {}
        for n in range(1, num + 1):
            choices[n] = input('Enter outcome of ' + str(n) + ': ')

        while want_to_die:
            val = roll(num)
            print(val)
            print(choices[val])
            ans = input('Want to play again? y or n: ')
            if ans != 'y':
                want_to_die = False

    def awareness(self):
        """
        1. needs memory
        2. can classify memory
        3. and adjust classifications
        4. creates output based on classifications
        """

    def train_camera(self, name):
        imagecount = 1000
        video = cv2.VideoCapture(0)
        count = 0
        d_valid = os.path.join(self.memory['camera_directory'], 'valid')
        d_train = os.path.join(self.memory['camera_directory'], 'training')
        final_v = os.path.join(d_valid, str(self.classification_number))
        final_t = os.path.join(d_train, str(self.classification_number))
        final_directory = os.path.join(self.memory['camera_directory'], str(self.classification_number))

        try:
            os.makedirs(final_directory)
        except FileExistsError:
            for file_name in os.listdir(final_directory):
                # construct full file path
                file = final_directory + file_name
                if os.path.isfile(file):
                    print('Deleting file:', file)
                    os.remove(file)

        while count < imagecount:
            count += 1
            _, frame = video.read()
            im = Image.fromarray(frame, 'RGB')
            im = im.resize((128, 128))

            im.save(os.path.join(final_directory, str(count) + ".jpg"), "JPEG")
            im.save(os.path.join(final_t, str(count) + ".jpg"), "JPEG")
            if count < 200:
                im.save(os.path.join(final_v, str(count) + ".jpg"), "JPEG")
            # cv2.imshow("Capturing", frame)
            # key = cv2.waitKey(1)
            # if key == ord('q'):
            #     break
        self.eyes.fit(self.train_generator, epochs=4, steps_per_epoch=63,
                                validation_data=self.validation_generator, validation_steps=7, workers=4)
        self.eyes.save('eyes.h5')
        num = int(self.classification_number)
        np = Person(name, num, im)
        self.memory['names'].append(name)
        self.new_memory("People", num, np)
        self.classification_number += 1
        video.release()

    def is_face_known(self, im):
        im = im.resize((128, 128))
        img_array = np.array(im)

        # Our keras model used a 4D tensor, (images x height x width x channel)
        # So changing dimension 128x128x3 into 1x128x128x3
        img_array = np.expand_dims(img_array, axis=0)

        # Calling the predict method on model to predict 'me' on the image
        prediction = self.eyes.predict(img_array)[0][0]
        if prediction in self.memory['People'].keys():
            return [prediction]
        return None

    def patience(self, prompt, timeout=5):
        start_time = time.time()
        user_input = ''
        while time.time() - start_time < timeout:
            user_input = input(prompt)
            if user_input:
                break
        return user_input

# @pyRAPL.measureit
def friend(life, name):
    # pyRAPL.setup()
    # meter = pyRAPL.Measurement('bar')
    # meter.begin()
    is_alive = life
    current_time = datetime.datetime.now()
    brain = Brain(
        name, current_time,
    )
    brain.vid = cv2.VideoCapture(0)

    print('-------------------------------------------------------------------')
    while is_alive:
        # Thread(target=brain.check).start()
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
                'camera_input': brain.vid.read(),  # [ 0. container [ 1. column [2. Row [ 3. rgb ] ] ] ]
            }

        response = brain.interpret(inputs)
        print(response[0])
        is_alive = brain.alive

        print('-------------------------------------------------------------------')
    if brain.ec:
        brain.vid.release()
        cv2.destroyAllWindows()
    # meter.end()


#
# class Chatbot:
#     def __init__(self):
#         self.data = {"intents": [
#
#             {"tag": "age",
#              "patterns": ["how old are you?"],
#              "responses": ["I am 2 years old and my birthday was yesterday"]
#              },
#             {"tag": "greeting",
#              "patterns": ["Hi", "Hello", "Hey"],
#              "responses": ["Hi there", "Hello", "Hi :)"],
#              },
#             {"tag": "goodbye",
#              "patterns": ["bye", "later"],
#              "responses": ["Bye", "take care"]
#              },
#             {"tag": "name",
#              "patterns": ["what's your name?", "who are you?"],
#              "responses": ["I have no name yet," "You can give me one, and I will appreciate it"]
#              }
#
#         ]}
#
#         self.lm = WordNetLemmatizer()  # for getting words
#         # lists
#         self.ourClasses = []
#         self.newWords = []
#         documentX = []
#         documentY = []
#         # Each intent is tokenized into words and the patterns and their associated tags are added to their respective lists.
#         for intent in self.data["intents"]:
#             for pattern in intent["patterns"]:
#                 ournewTkns = word_tokenize(pattern)  # tokenize the patterns
#                 self.newWords.extend(ournewTkns)  # extends the tokens
#                 documentX.append(pattern)
#                 documentY.append(intent["tag"])
#
#             if intent["tag"] not in self.ourClasses:  # add unexisting tags to their respective classes
#                 self.ourClasses.append(intent["tag"])
#
#         self.newWords = [self.lm.lemmatize(word.lower()) for word in self.newWords if
#                          word not in string.punctuation]  # set words to lowercase if not in punctuation
#         self.newWords = sorted(set(self.newWords))  # sorting words
#         self.ourClasses = sorted(set(self.ourClasses))  # sorting classes
#
#         trainingData = []  # training list array
#         outEmpty = [0] * len(self.ourClasses)
#         # bow model
#         for idx, doc in enumerate(documentX):
#             bagOfwords = []
#             text = self.lm.lemmatize(doc.lower())
#             for word in self.newWords:
#                 bagOfwords.append(1) if word in text else bagOfwords.append(0)
#
#             outputRow = list(outEmpty)
#             outputRow[self.ourClasses.index(documentY[idx])] = 1
#             trainingData.append([bagOfwords, outputRow])
#
#         random.shuffle(trainingData)
#         trainingData = np.array(trainingData, dtype=object)  # coverting our data into an array after shuffling
#
#         x = np.array(list(trainingData[:, 0]))  # first trainig phase
#         y = np.array(list(trainingData[:, 1]))  # second training phase
#
#         iShape = (len(x[0]),)
#         oShape = len(y[0])
#         # parameter definition
#         self.model = Sequential()
#         # In the case of a simple stack of layers, a Sequential model is appropriate
#
#         # Dense function adds an output layer
#         self.model.add(Dense(128, input_shape=iShape, activation="relu"))
#         # The activation function in a neural network is in charge of converting the node's summed weighted input into activation of the node or output for the input in question
#         self.model.add(Dropout(0.5))
#         # Dropout is used to enhance visual perception of input neurons
#         self.model.add(Dense(64, activation="relu"))
#         self.model.add(Dropout(0.3))
#         self.model.add(Dense(oShape, activation="softmax"))
#         # below is a callable that returns the value to be used with no arguments
#         md = tf.keras.optimizers.Adam(learning_rate=0.01, decay=1e-6)
#         # Below line improves the numerical stability and pushes the computation of the probability distribution into the categorical crossentropy loss function.
#         self.model.compile(loss='categorical_crossentropy',
#                            optimizer=md,
#                            metrics=["accuracy"])
#         # Output the model in summary
#         # print(self.model.summary())
#         # Whilst training your Nural Network, you have the option of making the output verbose or simple.
#         self.model.fit(x, y, epochs=200, verbose=1)
#         # By epochs, we mean the number of times you repeat a training set.
#
#     def ourText(self, text):
#         newtkns = word_tokenize(text)
#         newtkns = [self.lm.lemmatize(word) for word in newtkns]
#         return newtkns
#
#     def wordBag(self, text, vocab):
#         newtkns = self.ourText(text)
#         bagOwords = [0] * len(vocab)
#         for w in newtkns:
#             for idx, word in enumerate(vocab):
#                 if word == w:
#                     bagOwords[idx] = 1
#         return np.array(bagOwords)
#
#     def Pclass(self, text, vocab, labels):
#         bagOwords = self.wordBag(text, vocab)
#         ourResult = self.model.predict(np.array([bagOwords]))[0]
#         newThresh = 0.2
#         yp = [[idx, res] for idx, res in enumerate(ourResult) if res > newThresh]
#
#         yp.sort(key=lambda x: x[1], reverse=True)
#         newList = []
#         for r in yp:
#             newList.append(labels[r[0]])
#         return newList
#
#     def getRes(self, firstlist, fJson):
#         tag = firstlist[0]
#         listOfIntents = fJson["intents"]
#         for i in listOfIntents:
#             if i["tag"] == tag:
#                 ourResult = random.choice(i["responses"])
#                 break
#         return ourResult







if __name__ == '__main__':
    AI_is_alive = True
    friend(AI_is_alive, "Zoro")
