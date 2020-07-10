import numpy as np
import cv2

earCascade = cv2.CascadeClassifier('Cascades/haarcascade_mcs_rightear.xml')

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
        green = (0,255,0)
        scaling_h = 0.05
        scaling_w = 0.2
        start_w = int(w * scaling_w)
        start_h= int(h * scaling_h)
        stop_w = int(w * (1+scaling_w))
        stop_h = int(h * (1+scaling_h))
        cv2.rectangle(img, (x-start_w,y-start_h), (x+stop_w,y+stop_h), color=green, thickness=1)

        # roi_gray = gray[y:y+h, x:x+w]
        # roi_color = img[y:y+h, x:x+w]
        

    cv2.imshow('video',img)

    k = cv2.waitKey(30) & 0xff
    if k == 27: # press 'ESC' to quit
        break
    if k == ord('p'):
#         img = img[top+1:bottom, left+1:right] # +1 eliminates rectangle artifacts
#         # Re-flip image to original
        img = cv2.flip(img, 1)
        # Save the captured image into the datasets folder
        cv2.imwrite("test_im.png", img)
        print('Take image')

cap.release()
cv2.destroyAllWindows()