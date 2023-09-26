import numpy as np
import SpeechRecognition as sr


#
#
#  If it's a good AI I shouldn't have to train it before it's aware.
#
#
########################################################################################################################
############ Train model and adjust transformation matrix. use similarity score of inputs to adjust.
############ get eigenvectors of each input
############ Branches from inputs like tree

class Brain:
    class Neutrons:
        def __init__(self):
            connections = {}  # neighbors
            associations = {}

        def pathway(self):
            yeah = 9

    def __init__(self):
        """
        audio:
        - need audio to text? sp
        - text2num?

        camera:
        - 1280x720


        """



        #########
        # Audio #
        #########
        r = sr.Recognizer()
        mic = sr.Microphone()
        with mic as source:
            r.adjust_for_ambient_noise(source)
            audio = r.listen(source)
            transcript = r.recognize_google(audio)
            print(transcript)

        ##########
        # Camera #
        ##########

        ###################
        # internet access #
        ###################


    def txt2num(self, text):
        num = 0


if __name__ == '__main__':
    r = sr.Recognizer()
    mic = sr.Microphone()
    with mic as source:
        r.adjust_for_ambient_noise(source)
        audio = r.listen(source)
        transcript = r.recognize_google(audio)
        print(transcript)