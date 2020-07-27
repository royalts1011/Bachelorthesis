import cv2
import os
from os.path import join, dirname, exists
import time

#########################################################################
# SET PARAMETERS
#########################################################################

# set amount of pictures and pictures per step setting
PICTURES  = 80
STEP = 20
RECT_COL = (0,255,0)

DATASET_DIR = '../dataset'
PIC_DIR = '../temp_imgs'

# additional space around the ear to be captured
# 0.1 is tightly around, 0.2 more generous 
SCALING = 0.2
SCALING_H = 0.05
SCALING_W = 0.2 

INSTRUCTION = "\n [INFO] Initializing ear capture. Turn your head left. Your right ear should then be facing the camera."

#########################################################################

def make_720(object):
    object.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    object.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
def make_540(object):
    object.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    object.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
def make_480(object):
    object.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    object.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
def make_240(object):
    object.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    object.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

def rescale_frame(frame, percent=75):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)

def capture_ear_images(amount_pic=PICTURES, pic_per_stage=STEP, margin=SCALING, is_authentification=False):

    cap = cv2.VideoCapture(0)
    time.sleep(2.0)
    # open window dimensions
    make_720(cap)

    ear_detector = cv2.CascadeClassifier('Cascades/haarcascade_mcs_rightear.xml')

    # Set correct folder path and person's name
    target_folder = (DATASET_DIR, PIC_DIR)[is_authentification]
    ear_name = (input('\n Enter name end press <return> ==>  '), 'unknown')[is_authentification]   
    if not exists(target_folder): os.mkdir(target_folder)
    usr_dir = join(target_folder, ear_name)
    if not exists(usr_dir): os.mkdir(usr_dir)
    print(INSTRUCTION)

        
    # Initialize individual sampling ear count
    count = 0

    while True:
        # receive image
        ret, frame = cap.read()
        # flip video frame horizontally to show it "mirror-like"
        frame = cv2.flip(frame, 1)
        rects = ear_detector.detectMultiScale(frame, 1.1, 5)

        for (x,y,w,h) in rects:
            # bounding box will be bigger by increasing the scaling
            left = x - int(w * SCALING_W)
            top = y - int(h * SCALING_H)
            right = x + int(w * (1+SCALING_W))
            bottom = y + int(h * (1+SCALING_H))

            cv2.rectangle(frame, (left, top), (right, bottom), color=RECT_COL, thickness=1)   
            count += 1
            cv2.imshow('Frame', frame)

            frame = frame[top+1:bottom, left+1:right] # +1 eliminates rectangle artifacts
            # Re-flip image to original
            frame = cv2.flip(frame, 1)
            # Save the captured image into the datasets folder
            cv2.imwrite(join(usr_dir, (ear_name + "{0:0=3d}".format(count) + ".png")), frame)

            # display after defined set of steps 
            if (count%pic_per_stage) == 0 and count != amount_pic:
                print("\n [INFO] Next step commencing... \n")
                print(count)
                input("Reposition your head and press <return> to continue.")


        k = cv2.waitKey(100) & 0xff # Press 'ESC' for exiting video
        if k == 27:
            break
        elif count >= amount_pic: # Stop loop when the amount of pictures is collected
            print(count)
            break


    # Do a bit of cleanup
    print("\n [INFO] Exiting Program.")
    cap.release()
    cv2.destroyAllWindows()


if __name__=='__main__':
    capture_ear_images()    
