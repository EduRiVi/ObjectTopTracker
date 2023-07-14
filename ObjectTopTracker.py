from __future__ import print_function
import cv2 as cv
import numpy as np
import os


cap = cv.VideoCapture(0)


while True:
 
    ret, frame = cap.read()

    wd = frame.shape[0]
    he = frame.shape[1]

    if frame is None:
        break
    
    frame_HSV = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    frame_HSV = cv.inRange(frame_HSV, (35, 80, 30), (70, 255, 194))
 
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3,3))
    frame_HSV = cv.morphologyEx(frame_HSV, cv.MORPH_OPEN, kernel)
    frame_HSV = cv.GaussianBlur(frame_HSV, (3, 3), 10)
 
    contours, hier = cv.findContours(frame_HSV, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    lastWidth = 0
    elem = None
    
    if len(contours) > 0 :
        for cont in contours :
            rect = cv.minAreaRect(cont)
            width = min([rect[1][0], rect[1][1]])

            if lastWidth < width > 50:
                lastWidth = width
                elem = cont

        if elem is not None:

            rect = cv.minAreaRect(elem)
            box = cv.boxPoints(rect)
            box = np.int0(box)
            cv.drawContours(frame, [box], 0, (0, 0, 255), 2)
    
            if rect[1][0] > rect[1][1] :
                midPoint = (round((box[0][0] + box[1][0]) / 2), round((box[0][1] + box[1][1]) / 2))
            else :
                midPoint = (round((box[1][0] + box[2][0]) / 2), round((box[1][1] + box[2][1]) / 2))

            cv.circle(frame, midPoint, 5, (255, 0, 0), -1)

            cv.putText(frame, 'AREA: ' + str(rect[1][0] * rect[1][1]), (20,40), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0))
            cv.putText(frame, 'WIDTH: ' + str(min([rect[1][0], rect[1][1]])), (20,70), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0))

    
    cv.imshow('window_detection', frame)
 
    key = cv.waitKey(30)
    if key == ord('q') or key == 27:
        break

cap.release()
cv.destroyAllWindows()