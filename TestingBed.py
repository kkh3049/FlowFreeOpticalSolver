#!/usr/bin/python
import re

import cv2
import sys
import numpy as np
import pytesseract

PREVIEW = 0
CANNY = 1
GRID_BOUNDS = 2
CIRCLES = 3
STATUS_TEXT = 4


def findBiggestContour(contours):
    maxArea = 0
    cntIdx = -1
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > 1000:
            if area > maxArea:
                maxArea = area
                bestCnt = contour
                cntIdx = i
    return cntIdx, bestCnt


def resizeAndPad(img, size, padColor=0):
    h, w = img.shape[:2]
    sh, sw = size

    # interpolation method
    if h > sh or w > sw:  # shrinking image
        interp = cv2.INTER_AREA
    else:  # stretching image
        interp = cv2.INTER_CUBIC

    # aspect ratio of image

    wScale = sw / w
    hScale = sh / h

    if wScale > hScale:
        new_w = int(w * hScale)
        new_h = sh
        pad_horz = (sw - new_w) / 2
        pad_left, pad_right = np.floor(pad_horz).astype(int), np.ceil(pad_horz).astype(int)
        pad_top, pad_bot = 0, 0
    elif wScale < hScale:
        new_w = sw
        new_h = int(h * wScale)
        pad_vert = (sh - new_h) / 2
        pad_top, pad_bot = np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
        pad_left, pad_right = 0, 0
    else:  # aspectRatio matches
        new_h, new_w = sh, sw
        pad_left, pad_right, pad_top, pad_bot = 0, 0, 0, 0

    # set pad color
    if len(img.shape) == 3 and not isinstance(padColor,
                                              (list, tuple, np.ndarray)):
        padColor = [padColor] * 3

    # scale and pad
    scaled_img = cv2.resize(img, (new_w, new_h), interpolation=interp)
    scaled_img = cv2.copyMakeBorder(scaled_img, pad_top, pad_bot, pad_left, pad_right, borderType=cv2.BORDER_CONSTANT,
                                    value=padColor)

    return scaled_img


feature_params = dict(maxCorners=500,
                      qualityLevel=0.2,
                      minDistance=15,
                      blockSize=9)
s = 0
if len(sys.argv) > 1:
    s = sys.argv[1]

image_filter = PREVIEW
alive = True

win_name = 'Camera Filters'
cv2.namedWindow(win_name, flags=cv2.WINDOW_KEEPRATIO | cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_EXPANDED)
result = None

source = cv2.VideoCapture(s)
frame = cv2.imread(r'C:\Users\final\Pictures\gridTest.png')

uThresh = 150
sWindow = 3
accumThresh = 35

while alive:
    curFrame = frame.copy()

    if image_filter == PREVIEW:
        result = curFrame
    canny = cv2.Canny(curFrame, 80, uThresh, apertureSize=sWindow)
    if image_filter == CANNY:
        result = canny
    contours, hierarchy = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    biggestContourIndex, biggestContour = findBiggestContour(contours)
    cv2.drawContours(canny, contours, -1, (255, 255, 255), 3)
    contours, hierarchy = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    biggestContourIndex, biggestContour = findBiggestContour(contours)

    gridBounds = cv2.boundingRect(biggestContour)
    points = [gridBounds[0], gridBounds[1], gridBounds[0] + gridBounds[2], gridBounds[1] + gridBounds[3]]

    contourImage = np.stack((canny,) * 3, axis=-1)
    cv2.rectangle(contourImage, (points[0], points[1]),
                  (points[2], points[3]), (0, 0, 255), 2)
    if image_filter == GRID_BOUNDS:
        result = contourImage

    textStartFound = False
    for i in range(gridBounds[1] - 1):
        lineIndex = gridBounds[1] - i - 1
        lineHasWhite = 255 in canny[lineIndex, :]
        if textStartFound:
            if not lineHasWhite:
                textTop = lineIndex
                break
        else:
            if lineHasWhite:
                textBottom = lineIndex - 1
                textStartFound = True

    statusLineImg = curFrame[textTop:textBottom, points[0]:points[2]]
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    tessConfig = "-l eng --oem 1 --psm 7"
    statusText = pytesseract.image_to_string(statusLineImg, config=tessConfig)
    match = re.search(r'flows:\s*([0-9]+)/\s*([0-9]+)\s', statusText)

    curFlowCount = int(match.group(1))
    goalFlowCount = int(match.group(2))

    if image_filter == STATUS_TEXT:
        result = statusLineImg

    gridArea = cv2.cvtColor(curFrame[points[1]:points[3], points[0]:points[2]], cv2.COLOR_BGR2GRAY)
    minRadius = min(gridBounds[2], gridBounds[3]) // 31
    maxRadius = minRadius * 32 // 9
    circles = cv2.HoughCircles(gridArea, cv2.HOUGH_GRADIENT, 1, minRadius * 2, param1=150, param2=accumThresh,
                               minRadius=minRadius,
                               maxRadius=maxRadius)
    circleImage = np.stack((canny,) * 3, axis=-1)
    for circle in circles[0]:
        cv2.circle(circleImage, (int(circle[0] + gridBounds[0]), int(circle[1] + gridBounds[1])), int(circle[2]),
                   (0, 255, 255), 3)

    assert len(circles[0]) == (goalFlowCount * 2)

    if image_filter == CIRCLES:
        result = circleImage

    winRect = cv2.getWindowImageRect(win_name)
    resizedResult = resizeAndPad(result, (winRect[2], winRect[3]))
    if image_filter == STATUS_TEXT:
        cv2.putText(resizedResult, statusText, (10, 50),
                    cv2.FONT_HERSHEY_PLAIN,
                    3,
                    (255, 0, 0))

    cv2.imshow(win_name, resizedResult)

    key = cv2.waitKey(1)
    if key == ord('Q') or key == ord('q') or key == 27:
        alive = False
    elif key == ord('C') or key == ord('c'):
        image_filter = CANNY
    elif key == ord('G') or key == ord('g'):
        image_filter = GRID_BOUNDS
    elif key == ord('Z') or key == ord('z'):
        image_filter = CIRCLES
    elif key == ord('T') or key == ord('t'):
        image_filter = STATUS_TEXT
    elif key == ord('P') or key == ord('p'):
        image_filter = PREVIEW
    elif key == ord('1'):
        accumThresh -= 1
    elif key == ord('!'):
        accumThresh += 1
    elif key == ord('2'):
        uThresh -= 1
    elif key == ord('@'):
        uThresh += 1
    elif key == ord('3'):
        sWindow -= 2
    elif key == ord('#'):
        sWindow += 2

source.release()
cv2.destroyWindow(win_name)
