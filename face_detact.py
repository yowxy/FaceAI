import cv2 as cv

img = cv.imread("photos/image.png")

cv.imshow("person",img)

gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
# cv.imshow("gray",gray)

haar_casades = cv.CascadeClassifier('haar_face.xml')

face_react = haar_casades.detectMultiScale(gray,scaleFactor=1.1, minNeighbors=3)
print(f"Number of face found ={len(face_react)}")

for (x,y,w,h) in face_react:
    cv.rectangle(img, (x,y), (x+w,y+h), (0,255,0) , thickness=3)


cv.imshow("face detect", img)

cv.waitKey(0)