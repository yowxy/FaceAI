import cv2 as cv
import numpy as np


blank = np.zeros((500, 500,3), dtype='uint8')
cv.imshow("blank", blank)

img = cv.imread('photos/EL GASING.png')
# cv.imshow('ini emyu',img)

#paint bg
# blank[:] = 0,255,0
# cv.imshow("Green",blank)


#draw rectangle
# cv.rectangle(blank ,(0,0), (blank.shape[1]//2, blank.shape[0]//2), (255,0,0),  thickness=cv.FILLED)
# cv.imshow("Rectangle",blank)


# draw circle
cv.circle(blank,(blank.shape[1]//2, blank.shape[0]//2), 40 , (0,0,255), thickness=-1)
# cv.imshow("circle",blank)

#line
cv.line(blank,(200,300), (100,230), (255,255,255), thickness=3)
# cv.putText(blank,"hello my name is Iklil ",(0,255),cv.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
# cv.imshow('line',blank)

# canny edge


cv.waitKey(0)