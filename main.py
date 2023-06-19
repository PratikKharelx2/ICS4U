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

#importing libraries
#so many ;(
import cv2 as cv
from hand_tracking import hand_detector
from classification import main_func_classifier
import numpy as np
import math
from itertools import groupby
from response_generator import resp
import time

#setting video capture aswell as hand identifiers and sign interperator
cap = cv.VideoCapture(0)
detector = hand_detector(maxHands=1)
classifier = main_func_classifier("detection_model/keras_model.h5", "detection_model/labels.txt")

#image output variables
offset = 20
imgSize = 224
labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z","Done"]

#function for formatiing the input list into understandable text
def respond(text):
    min_count = 5
    string = []
    current_num = text[0]
    current_count = 0
    # clearing out all parts of the string that are below the limit.abs
    # the overall part of this code is this:
    # get an input list of [a,a,a,a,a,a,b,b,s,s,s,s,s,b,s,s,r,r,b,b,b,r,r]
    # we can see that the list has more than the needed 5 count
    # but since they are not in a row, the letter can be remove from that part.abs
    # this is to prevent some small glitches that sneak their way into the final input for the response
    # by filtering all the small particulates, we can have a clean input that the API can understand.
    for num in text:
        if num == current_num: current_count += 1
        else:
            if current_count >= min_count:
                string.extend([current_num] * current_count)
            current_num = num
            current_count = 1
    # if the count is above the required, add it to the list for response
    if current_count >= min_count: string.extend([current_num] * current_count)
    #remove all duplicates that are in a row
    user_input = ''.join(char for char, _ in groupby(string))
    response = resp(user_input)
    # print the response from OpenAI
    print(f'Input:{user_input} \nResponse:{response}')
    #wait a bit before re-taking input
    time.sleep(3)

#main function that gets video input and runs it through everything
def main():
    # list with letters initialize
    text_list = []
    while True:
        #take camera input and modify
        success, img = cap.read()
        img = cv.flip(img,1)
        imgOutput = img.copy()
        #check if there are hands in the image
        hands, img = detector.findHands(img)
        #if there are hands, we can do the following:
        if hands:
            # get the first hand
            hand = hands[0]
            # get the x,y cords of the hand as well as the max width and max height
            x, y, w, h = hand['bbox']

            #generate an image array
            #bassically it's just a list, but because of its 3 val list, we can interperate it as RGB and turn it into an image
            #this particular image is made 224x224(which is the requirement for the hand classification function) and the image is made white with the *255
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            #crop the image
            imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

            #we need to get the shape that the crop is
            imgCropShape = imgCrop.shape

            #the aspect ratio makes it easier to write and understand the code in future writing
            aspectRatio = h / w

            #this part of the code is for getting only the haands for the classifier
            #it works with different ratios so that in the end we have a image output that is 224x224 no matter what the orientation

            # if the height is bigger than the width
            if aspectRatio > 1: 
                # just figure out the parts of the image shapes we need
                k = imgSize / h 
                wCal = math.ceil(k * w)
                imgResize = cv.resize(imgCrop, (wCal, imgSize))
                #resize the shape accordingly
                imgResizeShape = imgResize.shape
                #calc the gap between the edges
                wGap = math.ceil((imgSize - wCal) / 2)
                #resize the final image
                imgWhite[:, wGap:wCal + wGap] = imgResize
                #send the final image to the classifier
                #the classifier will return it's prediction as well as the index for the shape that it's predicting
                prediction, index = classifier.getPrediction(imgWhite, draw=False)
                #if the index is the 'Done' index
                if index == 26:
                    #we can send the input to the response function and clear the list right after
                    respond(text_list)
                    text_list = []
                    
            else:
                # same here: figure out the parts of the image shapes we need
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv.resize(imgCrop, (imgSize, hCal))
                #resize the shape accordingly
                imgResizeShape = imgResize.shape
                #calc the gap between the edges
                hGap = math.ceil((imgSize - hCal) / 2)
                #resize the final image
                imgWhite[hGap:hCal + hGap, :] = imgResize
                #send the final image to the classifier
                #the classifier will return it's prediction as well as the index for the shape that it's predicting
                prediction, index = classifier.getPrediction(imgWhite, draw=False)
                #if the index is the 'Done' index
                if index == 26:
                    #we can send the input to the response function and clear the list right after
                    respond(text_list)
                    text_list = []
            
            #once we get the prediction, we can add the letter to the final result
            text_list.append(labels[index])

            #adding the what shape it is on the top of the hand
            cv.rectangle(imgOutput, (x - offset, y - offset-50),
                        (x - offset+90, y - offset-50+50), (0, 0, 0), cv.FILLED)
            cv.putText(imgOutput, labels[index], (x, y -26), cv.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
        # if the user's hand is off the screen, it is counted as a space
        else: 
            text_list.append(' ')
        # output the image
        cv.imshow("Sign Language Detector", imgOutput)
        key = cv.waitKey(1)
        # if the exit key is pressed
        if key == 27:
            #leave
            break
    return#leave
main() # run main
cv.destroyAllWindows() # if it's left, close all windows
