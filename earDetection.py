import numpy as np
import cv2

earCascade = cv2.CascadeClassifier('Cascades/haarcascade_mcs_rightear.xml')

#########################################################################
# SET PARAMETERS
#########################################################################

GREEN = (0,255,0)

# additional space around the ear to be captured
# 0.1 is tightly around, 0.2 more generous 
SCALING = 0.2
SCALING_H = 0.05
SCALING_W = 0.2 

#########################################################################


cap = cv2.VideoCapture(0)
# open window dimensions
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280) # set Width
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720) # set Height

while True:
    # ignore boolean return Value, only receive image
    ret, img = cap.read()
    # flip video frame horizontally as webcams take mirror image
    img = cv2.flip(img, 1) 
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ears = earCascade.detectMultiScale(
        grey,
        scaleFactor=1.1,
        minNeighbors=5,     
        minSize=(20, 20)
    )

    for (x,y,w,h) in ears:        
        # bounding box will be bigger by increasing the scaling
        left = x - int(w * SCALING_W)
        top = y - int(h * SCALING_H)
        right = x + int(w * (1+SCALING_W))
        bottom = y + int(h * (1+SCALING_H))
        cv2.rectangle(img, (left, top), (right, bottom), color=GREEN, thickness=1)

    cv2.imshow('video',img)

    k = cv2.waitKey(30) & 0xff
    if k == 27: # press 'ESC' to quit
        break
    if k == ord('p'):
        img = img[top+1:bottom, left+1:right] # +1 eliminates rectangle artifacts
        # Re-flip image to original
        img = cv2.flip(img, 1)
        # Save the captured image into the datasets folder
        cv2.imwrite("../SIZE_TEST.png", img)
        print('Test image taken')

cap.release()
cv2.destroyAllWindows()