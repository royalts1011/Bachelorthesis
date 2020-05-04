import numpy as np
import cv2

earCascade = cv2.CascadeClassifier('Cascades/haarcascade_mcs_rightear.xml')

cap = cv2.VideoCapture(0)
# open window dimensions
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640) # set Width
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480) # set Height

while True:
    # ignore boolean return Value, only receive image
    ret, img = cap.read()
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
        green = (0,255,0)
        scaling = 0.2
        start_w = int(w * scaling)
        start_h= int(h * scaling)
        stop_w = int(w * (1+scaling))
        stop_h = int(h * (1+scaling))
        cv2.rectangle(img, (x-start_w,y-start_h), (x+stop_w,y+stop_h), color=green, thickness=1)

        # roi_gray = gray[y:y+h, x:x+w]
        # roi_color = img[y:y+h, x:x+w]
        

    cv2.imshow('video',img)

    k = cv2.waitKey(30) & 0xff
    if k == 27: # press 'ESC' to quit
        break

cap.release()
cv2.destroyAllWindows()