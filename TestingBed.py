#!/usr/bin/python
import re
from collections import defaultdict
from itertools import tee, islice, chain

import cv2
import sys
import numpy as np
import pytesseract

import FlowSolver

PREVIEW = 0
CANNY = 1
GRID_BOUNDS = 2
CIRCLES = 3
STATUS_TEXT = 4
TOP_TEXT = 5
DATA = 6
SOLVED = 7

def iterate_with_next(iterable):
    items, nexts = tee(iterable, 2)
    nexts = chain(islice(nexts, 1, None), [None])
    return zip(items, nexts)

def find_biggest_contour(contours):
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


def find_start_and_end_of_white(img, startLine, searchReverse=False, searchRows=True):
    textStartFound = False
    firstWhite = -1
    changeToNonWhite = -1
    numLines = img.shape[0] if searchRows else img.shape[1]
    numLinesToSearch = numLines - startLine if not searchReverse else startLine
    for i in range(numLinesToSearch):
        lineIndex = startLine + i if not searchReverse else startLine - i
        lineHasWhite = (255 in img[lineIndex, :]) if searchRows else (255 in img[:, lineIndex])
        if textStartFound:
            if not lineHasWhite:
                changeToNonWhite = lineIndex
                break
        else:
            if lineHasWhite:
                firstWhite = lineIndex
                textStartFound = True
    if not searchReverse:
        return firstWhite, changeToNonWhite
    else:
        return changeToNonWhite, firstWhite


def resize_and_pad(img, size, padColor=0):
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

win_name = 'Flow Solver'
cv2.namedWindow(win_name, flags=cv2.WINDOW_KEEPRATIO | cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_EXPANDED)
result = None

source = cv2.VideoCapture(s)
frame = cv2.imread(r'C:\Users\final\Pictures\gridTest.png')

uThresh = 150
sWindow = 3
accumThresh = 35
ttThresh = 50

curFrame = frame.copy()

canny = cv2.Canny(curFrame, 80, uThresh, apertureSize=sWindow)
contours, hierarchy = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
biggestContourIndex, biggestContour = find_biggest_contour(contours)
cv2.drawContours(canny, contours, -1, (255, 255, 255), 3)
contours, hierarchy = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
biggestContourIndex, biggestContour = find_biggest_contour(contours)

gridBounds = cv2.boundingRect(biggestContour)
gridCorners = [gridBounds[0], gridBounds[1], gridBounds[0] + gridBounds[2], gridBounds[1] + gridBounds[3]]

textTop, textBottom = find_start_and_end_of_white(canny, gridBounds[1] - 1, searchReverse=True)
statusLineImg = curFrame[textTop - 1:textBottom + 1, gridCorners[0]:gridCorners[2]]
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
tessConfig = "-l eng --oem 1 --psm 7"
statusText = pytesseract.image_to_string(statusLineImg, config=tessConfig)
match = re.search(r'flows:\s*([0-9]+)/\s*([0-9]+)\s', statusText)

curFlowCount = int(match.group(1))
goalFlowCount = int(match.group(2))

topTextTop, topTextBottom = find_start_and_end_of_white(canny, 0)
topTextImg = curFrame[topTextTop - 1:topTextBottom + 1, gridCorners[0]:gridCorners[2]]
leftCircleLeft, leftCircleRight = find_start_and_end_of_white(topTextImg, 0, searchRows=False)
rightCircleLeft, rightCircleRight = find_start_and_end_of_white(topTextImg, topTextImg.shape[1] - 1,
                                                                searchReverse=True,
                                                                searchRows=False)
topTextImg = topTextImg[:, leftCircleRight + 5:rightCircleLeft - 5]
topTextImg = cv2.cvtColor(topTextImg, cv2.COLOR_BGR2GRAY)
thresh, topTextImg = cv2.threshold(topTextImg, ttThresh, 255, cv2.THRESH_BINARY)
tessConfig = "-l eng --oem 1 --psm 13"
topText = pytesseract.image_to_string(topTextImg, config=tessConfig)
match = re.search(r'level\s*([0-9]+)\s*([0-9]+)x([0-9]+)', topText)

curLevel = int(match.group(1))
curLevelWidth = int(match.group(2))
curLevelHeight = int(match.group(3))

squareWidth = gridBounds[2] / curLevelWidth
squareHeight = gridBounds[3] / curLevelHeight
squareCenters = [[(int(gridBounds[0] + (x + 0.5) * squareWidth), int(gridBounds[1] + (y + 0.5) * squareHeight)) for x in
                  range(curLevelWidth)] for y in range(curLevelHeight)]
contourImage = np.stack((canny,) * 3, axis=-1)

for row in squareCenters:
    for squareCenter in row:
        cv2.rectangle(contourImage,
                      (int(squareCenter[0] - squareWidth * 0.5), int(squareCenter[1] - squareHeight * 0.5)), (
                          int(squareCenter[0] + squareWidth * 0.5), int(squareCenter[1] + squareWidth * 0.5)),
                      (0, 0,
                       255), 3)
        cv2.circle(contourImage, (int(squareCenter[0]), int(squareCenter[1])), 5, (0, 255, 255), -1)

gridArea = cv2.cvtColor(curFrame[gridCorners[1]:gridCorners[3], gridCorners[0]:gridCorners[2]], cv2.COLOR_BGR2GRAY)
minRadius = min(gridBounds[2], gridBounds[3]) // 31
maxRadius = minRadius * 32 // 9
circles = cv2.HoughCircles(gridArea, cv2.HOUGH_GRADIENT, 1, minRadius * 2, param1=150, param2=accumThresh,
                           minRadius=minRadius,
                           maxRadius=maxRadius)

circles = [[(int(circle[0] + gridBounds[0]), int(circle[1] + gridBounds[1])), int(circle[2])] for circle in
           circles[0]]
assert len(circles) == (goalFlowCount * 2)

circles = [[circle[0], circle[1], curFrame[circle[0][1]][circle[0][0]]] for circle in circles]
circles = [[circle[0], circle[1], (int(circle[2][0]), int(circle[2][1]), int(circle[2][2]))] for circle in circles]
indexAndColors = [(idx, val[2]) for idx, val in enumerate(circles)]
colorIndexToCircleIndices = defaultdict(list)
colors = []

for _ in range(goalFlowCount):
    idx, color = indexAndColors[0]
    closestIndex = min(range(len(indexAndColors)),
                       key=lambda l_idx: 256 * 3 if l_idx == 0 else abs(
                           indexAndColors[l_idx][1][0] - color[0]) + abs(
                           indexAndColors[l_idx][1][1] - color[1]) + abs(indexAndColors[l_idx][1][2] - color[2]))
    oIdx, oColor = indexAndColors[closestIndex]
    avgColor = ((color[0] + oColor[0]) // 2, (color[1] + oColor[1]) // 2, (color[2] + oColor[2]) // 2)
    colorIndexToCircleIndices[len(colors)].append(idx)
    colorIndexToCircleIndices[len(colors)].append(oIdx)
    colors.append(avgColor)
    indexAndColors.remove((idx, color))
    indexAndColors.remove((oIdx, oColor))

assert len(colorIndexToCircleIndices) == goalFlowCount
for color, circleIndices in colorIndexToCircleIndices.items():
    assert len(circleIndices) == 2

circleImage = np.stack((canny,) * 3, axis=-1)
for circle in circles:
    cv2.circle(circleImage, circle[0], circle[1],
               circle[2], 3)
for colorIdx, circleIndices in colorIndexToCircleIndices.items():
    cv2.line(circleImage, circles[circleIndices[0]][0], circles[circleIndices[1]][0], colors[colorIdx], thickness=3)

circleSquareIndices = [
    (int((circle[0][0] - gridBounds[0]) / squareWidth), int((circle[0][1] - gridBounds[1]) / squareHeight)) for
    circle in circles]

gridData = [[(-1, False) for square in row] for row in squareCenters]
for colorIdx, circleIndices in colorIndexToCircleIndices.items():
    for circleIndex in circleIndices:
        xCircle, yCircle = circleSquareIndices[circleIndex]
        gridData[xCircle][yCircle] = (colorIdx, True)

startingDataImage = np.zeros(curFrame.shape, np.uint8)
for y, row in enumerate(gridData):
    for x, (squareColorIndex, squareIsCircle) in enumerate(row):
        if squareIsCircle:
            assert squareColorIndex >= 0
            cv2.circle(startingDataImage, squareCenters[x][y], minRadius, colors[squareColorIndex], -1)
        cv2.circle(startingDataImage, squareCenters[x][y], 10, (0, 255, 255), -1)
for colorIndex, circleIndices in colorIndexToCircleIndices.items():
    circleCenters = [squareCenters[circleSquareIndices[index][1]][circleSquareIndices[index][0]] for index in
                     circleIndices]
    cv2.line(startingDataImage, circleCenters[0], circleCenters[1], colors[colorIndex], 3)

solvedBoard, solvedPaths = FlowSolver.Solve(gridData)

solvedImage = curFrame.copy();
for colorIndex, path in enumerate(solvedPaths):
    color = colors[colorIndex]
    for cur, next in iterate_with_next(path):
        if next is None:
            continue
        startPoint = squareCenters[cur[1]][cur[0]]
        endPoint = squareCenters[next[1]][next[0]]
        cv2.line(solvedImage, startPoint, endPoint, color, 40)



while alive:
    if image_filter == PREVIEW:
        result = curFrame.copy()
    if image_filter == CANNY:
        result = canny.copy()
    if image_filter == GRID_BOUNDS:
        result = contourImage.copy()
    if image_filter == CIRCLES:
        result = circleImage.copy()
    if image_filter == STATUS_TEXT:
        result = statusLineImg.copy()
    if image_filter == TOP_TEXT:
        result = topTextImg.copy()
    if image_filter == DATA:
        result = startingDataImage
    if image_filter == SOLVED:
        result = solvedImage

    winRect = cv2.getWindowImageRect(win_name)
    resizedResult = resize_and_pad(result, (winRect[2], winRect[3]))
    if image_filter == STATUS_TEXT:
        cv2.putText(resizedResult, "flows:" + str(curFlowCount) + "/" + str(goalFlowCount),
                    (10, 120),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL,
                    2,
                    (255, 0, 0))
    elif image_filter == TOP_TEXT:
        cv2.putText(resizedResult, "level " + str(curLevel) + ": " + str(curLevelWidth) + "x" + str(curLevelHeight),
                    (10, 120),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL,
                    2,
                    (255, 0, 0))
    elif image_filter == CIRCLES:
        cv2.putText(resizedResult, "found " + str(len(circles)) + " circles",
                    (10, 120),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL,
                    2,
                    (255, 0, 0))

    cv2.imshow(win_name, resizedResult)

    key = cv2.waitKey(1)
    if key == ord('Q') or key == ord('q') or key == 27:
        alive = False
    elif key == ord('P') or key == ord('p') or key == ord('0'):
        image_filter = PREVIEW
    elif key == ord('C') or key == ord('c') or key == ord('1'):
        image_filter = CANNY
    elif key == ord('G') or key == ord('g') or key == ord('2'):
        image_filter = GRID_BOUNDS
    elif key == ord('Z') or key == ord('z') or key == ord('3'):
        image_filter = CIRCLES
    elif key == ord('T') or key == ord('t') or key == ord('4'):
        image_filter = STATUS_TEXT
    elif key == ord('Y') or key == ord('y') or key == ord('5'):
        image_filter = TOP_TEXT
    elif key == ord('D') or key == ord('d') or key == ord('6'):
        image_filter = DATA
    elif key == ord('S') or key == ord('s') or key == ord('7'):
        image_filter = SOLVED

source.release()
cv2.destroyWindow(win_name)
