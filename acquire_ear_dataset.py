import cv2
import os
# import copy

cam = cv2.VideoCapture(0)
# open window dimensions
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640) # set Width
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480) # set Height

ear_detector = cv2.CascadeClassifier('Cascades/haarcascade_mcs_rightear.xml')

# For each person, enter one numeric face id
ear_name = input('\n enter username end press <return> ==>  ')

print("\n [INFO] Initializing face capture. Look the camera and wait ...")
# Initialize individual sampling face count
count = 0

while(True):
    # ignore boolean return Value, only receive image
    ret, img = cam.read()
    # flip video frame horizontally as webcams take mirror image
    img = cv2.flip(img, 1)
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img_show = copy.copy(img)
    ears = ear_detector.detectMultiScale(img, 1.1, 5)

    

    for (x,y,w,h) in ears:
        green = (0,255,0)
        scaling = 0.2
        start_w = int(w * scaling)
        start_h= int(h * scaling)
        stop_w = int(w * (1+scaling))
        stop_h = int(h * (1+scaling))
        cv2.rectangle(img, (x-start_w,y-start_h), (x+stop_w,y+stop_h), color=green, thickness=1)   
        count += 1

        # Save the captured image into the datasets folder
        cv2.imwrite("dataset/User." + ear_name + '.' + str(count) + ".jpg", img[y-start_h+1:y+stop_h, x-start_w+1:x+stop_w]) # +1 eliminates rectangle artifacts

        cv2.imshow('image', img)

    k = cv2.waitKey(100) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break
    elif count >= 10: # Take 30 face sample and stop video
         break

# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()