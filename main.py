import cv2
import mediapipe as mp
import time
import pandas as pd
import numpy as np
import tensorflow as tf
interpreter = tf.lite.Interpreter(model_path="model1.tflite")

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

interpreter.allocate_tensors()

# input details
print(input_details[0]['shape'])
# output details
print(output_details)

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

while True :
    success,img = cap.read()
    imgRBG= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(imgRBG)
    cv2.line(img,(320,0),(320,(img.shape[0])),(0, 0, 0xFF), 8)
    # print(result.multi_hand_landmarks)
    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            mpDraw.draw_landmarks(img,handLms,mpHands.HAND_CONNECTIONS)
            # print([handLms.landmark])
            loc = np.zeros((1,21,6))
            for id , lm in enumerate(handLms.landmark):
                # if id==0 and lm.x > 0.45:
                #     print("left")
                #
                # if id==0 and lm.x < 0.55:
                #     print("right")
                if lm.x > 0.45:
                    # print("left",lm.x)
                    loc[0][id][0] = lm.x
                    loc[0][id][1] = lm.y
                    loc[0][id][2] = lm.z

                if lm.x < 0.55:
                    # print("right")
                    loc[0][id][3] = lm.x
                    loc[0][id][4] = lm.y
                    loc[0][id][5] = lm.z
        interpreter.set_tensor(input_details[0]['index'],loc.astype(np.float32))
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        i=0
        a=0
        r=-1
        # print(output_data)
        for lol in output_data:
            # print(max(lol))
            # print(min(lol))
            for x in lol:
                if x>a:
                    a=x
                    r=i
                i=i+1
        list =['TV',
         'after',
         'airplane',
         'all',
         'alligator',
         'animal',
         'another',
         'any',
         'apple',
         'arm',
         'aunt',
         'awake',
         'backyard',
         'bad',
         'balloon',
         'bath',
         'because',
         'bed',
         'bedroom',
         'bee',
         'before',
         'beside',
         'better',
         'bird',
         'black',
         'blow',
         'blue',
         'boat',
         'book',
         'boy',
         'brother',
         'brown',
         'bug',
         'bye',
         'callonphone',
         'can',
         'car',
         'carrot',
         'cat',
         'cereal',
         'chair',
         'cheek',
         'child',
         'chin',
         'chocolate',
         'clean',
         'close',
         'closet',
         'cloud',
         'clown',
         'cow',
         'cowboy',
         'cry',
         'cut',
         'cute',
         'dad',
         'dance',
         'dirty',
         'dog',
         'doll',
         'donkey',
         'down',
         'drawer',
         'drink',
         'drop',
         'dry',
         'dryer',
         'duck',
         'ear',
         'elephant',
         'empty',
         'every',
         'eye',
         'face',
         'fall',
         'farm',
         'fast',
         'feet',
         'find',
         'fine',
         'finger',
         'finish',
         'fireman',
         'first',
         'fish',
         'flag',
         'flower',
         'food',
         'for',
         'frenchfries',
         'frog',
         'garbage',
         'gift',
         'giraffe',
         'girl',
         'give',
         'glasswindow',
         'go',
         'goose',
         'grandma',
         'grandpa',
         'grass',
         'green',
         'gum',
         'hair',
         'happy',
         'hat',
         'hate',
         'have',
         'haveto',
         'head',
         'hear',
         'helicopter',
         'hello',
         'hen',
         'hesheit',
         'hide',
         'high',
         'home',
         'horse',
         'hot',
         'hungry',
         'icecream',
         'if',
         'into',
         'jacket',
         'jeans',
         'jump',
         'kiss',
         'kitty',
         'lamp',
         'later',
         'like',
         'lion',
         'lips',
         'listen',
         'look',
         'loud',
         'mad',
         'make',
         'man',
         'many',
         'milk',
         'minemy',
         'mitten',
         'mom',
         'moon',
         'morning',
         'mouse',
         'mouth',
         'nap',
         'napkin',
         'night',
         'no',
         'noisy',
         'nose',
         'not',
         'now',
         'nuts',
         'old',
         'on',
         'open',
         'orange',
         'outside',
         'owie',
         'owl',
         'pajamas',
         'pen',
         'pencil',
         'penny',
         'person',
         'pig',
         'pizza',
         'please',
         'police',
         'pool',
         'potty',
         'pretend',
         'pretty',
         'puppy',
         'puzzle',
         'quiet',
         'radio',
         'rain',
         'read',
         'red',
         'refrigerator',
         'ride',
         'room',
         'sad',
         'same',
         'say',
         'scissors',
         'see',
         'shhh',
         'shirt',
         'shoe',
         'shower',
         'sick',
         'sleep',
         'sleepy',
         'smile',
         'snack',
         'snow',
         'stairs',
         'stay',
         'sticky',
         'store',
         'story',
         'stuck',
         'sun',
         'table',
         'talk',
         'taste',
         'thankyou',
         'that',
         'there',
         'think',
         'thirsty',
         'tiger',
         'time',
         'tomorrow',
         'tongue',
         'tooth',
         'toothbrush',
         'touch',
         'toy',
         'tree',
         'uncle',
         'underwear',
         'up',
         'vacuum',
         'wait',
         'wake',
         'water',
         'wet',
         'weus',
         'where',
         'white',
         'who',
         'why',
         'will',
         'wolf',
         'yellow',
         'yes',
         'yesterday',
         'yourself',
         'yucky',
         'zebra']
        print (list[r])
        cv2.putText(img, (list[r]), (25, 15),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 0),2,cv2.LINE_AA)

    cv2.imshow("Image",img)
    cv2.waitKey(1)