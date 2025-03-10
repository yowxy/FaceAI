import cv2 as cv

img = cv.imread('photos/EL GASING.png')
# cv.imshow('GOAT',img)

#gray
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# cv.imshow('gray',gray)

#blur

blur = cv.GaussianBlur(img,(3,3), cv.BORDER_DEFAULT)
# cv.imshow('blur', blur)

# edge
canny = cv.Canny(img ,225,225)
cv.imshow('edge',canny)

# dilated
dilated = cv.dilate(canny, (3,3), iterations=1)
cv.imshow('dilated', dilated)


cv.waitKey(0)