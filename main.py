import cv2
import numpy as np

detect = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')

#cam = cv2.VideoCapture('D:\phone_folder\photos\Instagram\IMG_20230204_183137_801.jpg')
#cam = cv2.VideoCapture('D:\deep_learning\image2.png')
cam = cv2.VideoCapture('D:\deep_learning\Image.png')
check, img = cam.read()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = detect.detectMultiScale(gray,1.2,5)


for (x,y,w,h) in faces:
      cv2.rectangle(img, (x,y), (x+w, y+h), (255, 0, 0), 2)

cv2.imshow('Face Detect', img)
cv2.waitKey(600000)

cam.release()
cv2.destroyAllWindows()