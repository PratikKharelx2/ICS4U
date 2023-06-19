'''
Code written by Pratik Kharel
For Final Project in Course: ICS4UI
Main Function of the code:
-Online Talk Assistant(with slightly altered ASL). 
-Built-in hand detection that recognizes ASL to understand input from mute/deaf users. 
-Generates responses, does simple tasks, and chats with the user.
-Simple tasks include answering questions, keeping company, providing a shared AI experience for everyone.
*Program is limited in terms of processing power and input limitations due to the useage of Open AI's API.
This company has a free trial of their API which allows the code to work for a set amount of runs. 
Getting more inputs will require the use of real-world currency which, for me, is limited.
Future implementations will have no limit once a stable system for responses is added.*
'''

import cv2 as cv
from hand_tracking import hand_detector
from classification import main_func_classifier
import numpy as np
import math
from itertools import groupby
from response_generator import resp
import time

cap = cv.VideoCapture(0)
detector = hand_detector(maxHands=1)
classifier = main_func_classifier("detection_model/keras_model.h5", "detection_model/labels.txt")

offset = 20
imgSize = 224

text_list = []
labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z","Done"]

def respond(text):
    min_count = 6
    string = []
    current_num = text[0]
    current_count = 0
    for num in text:
        if num == current_num: current_count += 1
        else:
            if current_count >= min_count:
                string.extend([current_num] * current_count)
            current_num = num
            current_count = 1
    if current_count >= min_count: string.extend([current_num] * current_count)
    user_input = ''.join(char for char, _ in groupby(string))
    response = resp(user_input)
    print(f'Input: {user_input} \n Response: {response}')
    time.sleep(3)

def main():
    while True:
        success, img = cap.read()
        img = cv.flip(img,1)
        imgOutput = img.copy()
        hands, img = detector.findHands(img)
        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']

            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

            imgCropShape = imgCrop.shape

            aspectRatio = h / w

            if aspectRatio > 1:
                try:
                    k = imgSize / h
                    wCal = math.ceil(k * w)
                    imgResize = cv.resize(imgCrop, (wCal, imgSize))
                    imgResizeShape = imgResize.shape
                    wGap = math.ceil((imgSize - wCal) / 2)
                    imgWhite[:, wGap:wCal + wGap] = imgResize
                    prediction, index = classifier.getPrediction(imgWhite, draw=False)
                    if index == 26:
                        respond(text_list)
                except:
                    continue

            else:
                try:
                    k = imgSize / w
                    hCal = math.ceil(k * h)
                    imgResize = cv.resize(imgCrop, (imgSize, hCal))
                    imgResizeShape = imgResize.shape
                    hGap = math.ceil((imgSize - hCal) / 2)
                    imgWhite[hGap:hCal + hGap, :] = imgResize
                    prediction, index = classifier.getPrediction(imgWhite, draw=False)
                    if index == 26:
                        respond(text_list)
                except:
                    continue
            
            text_list.append(labels[index])

            cv.rectangle(imgOutput, (x - offset, y - offset-50),
                        (x - offset+90, y - offset-50+50), (0, 0, 0), cv.FILLED)
            cv.putText(imgOutput, labels[index], (x, y -26), cv.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
            cv.rectangle(imgOutput, (x-offset, y-offset),
                        (x + w+offset, y + h+offset), (0, 0, 0), 4)

        cv.imshow("Image", imgOutput)
        key = cv.waitKey(1)
        if key == 27:
            break
    return

main()
cv.destroyAllWindows()
