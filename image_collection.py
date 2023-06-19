import cv2 as cv
from hand_tracking import HandDetector
import numpy as np
import math
import time

cap = cv.VideoCapture(0)
detector = HandDetector(maxHands=1)

offset = 20
imgSize = 224
counter = 0
folder = "data_sets/A"

while True:
    success, img = cap.read()
    img = cv.flip(img,1)
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
            except:
                continue
        cv.imshow("ImageWhite", imgWhite)

    cv.imshow("Image", img)
    key = cv.waitKey(1)
    if key == ord("s") or key == ord("l"):
        if counter < 1000:
            cv.imwrite(f'{folder}/Image_{time.time()}.jpg',imgWhite)
            counter += 1
            print(counter)
        else: break
    if key == 27:
        break
    
cv.destroyAllWindows()