import numpy as np
import cv2

earCascade = cv2.CascadeClassifier('Cascades/haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# open window dimensions
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640) # set Width
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480) # set Height

while True:
    # ignore boolean return Value, only receive image
    _, img = cap.read()
    # flip video frame horizontally as webcams take mirror image
    img = cv2.flip(img, 1) 
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ears = earCascade.detectMultiScale(
        img,
        scaleFactor=1.1,
        minNeighbors=5,     
        minSize=(20, 20)
    )

    for (x,y,w,h) in ears:
        blue = (255,0,0)
        cv2.rectangle(img, (x,y), (x+w,y+h), color=blue, thickness=1)

        # roi_gray = gray[y:y+h, x:x+w]
        # roi_color = img[y:y+h, x:x+w]
        

    cv2.imshow('video',img)

    k = cv2.waitKey(30) & 0xff
    if k == 27: # press 'ESC' to quit
        break

cap.release()
cv2.destroyAllWindows()