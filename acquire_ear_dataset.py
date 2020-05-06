import cv2
import os
from playsound import playsound

cam = cv2.VideoCapture(0)
# open window dimensions
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640) # set Width
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480) # set Height

ear_detector = cv2.CascadeClassifier('Cascades/haarcascade_mcs_rightear.xml')

# For each person, enter a new identification name
ear_name = input('\n Enter username end press <return> ==>  ')
is_other = input('\n Should the user be part of the general dataset? (y/n)')

usr_dir = "dataset/" + (ear_name, "ext_group")[is_other.lower() == "y"]

if not os.path.exists(usr_dir):
    os.mkdir(usr_dir)

#########################################################################
# SET PARAMETERS
#########################################################################

PLAYSOUND = False

# set amount of pictures and pictures per head setting
amount_pictures  = 20
steps_of = 20

# additional space around the ear to be captured
# 0.1 is tightly around, 0.2 more generous 
scaling = 0.2

user_instructions = ["\n [INFO] Initializing ear capture. Turn your head left. Your right ear should then be facing the camera.", 
                "Look into the camera and slowly turn your head 45 degrees to the left",
                "Now look up, keeping the right ear towards the camera.",
                "Now look down, keeping the right ear towards the camera."
                ]
#########################################################################
# assert amount_pictures/10 <= (len(user_instructions))

print(user_instructions[0])

# Initialize individual sampling ear count
count = 0

while(True):
    # receive image
    ret, img = cam.read()
    # flip video frame horizontally as webcams take mirror image
    img = cv2.flip(img, 1)
    ears = ear_detector.detectMultiScale(img, 1.1, 5)

    for (x,y,w,h) in ears:
        # bounding box will be bigger by increasing the scaling
        start_w = int(w * scaling)
        start_h= int(h * scaling)
        stop_w = int(w * (1+scaling))
        stop_h = int(h * (1+scaling))
        green = (0,255,0)
        cv2.rectangle(img, (x-start_w,y-start_h), (x+stop_w,y+stop_h), color=green, thickness=1)   
        count += 1

        # Save the captured image into the datasets folder
        cv2.imwrite(usr_dir + "/" + ear_name +  "{0:0=3d}".format(count) + ".png", img[y-start_h+1:y+stop_h, x-start_w+1:x+stop_w]) # +1 eliminates rectangle artifacts
        cv2.imshow('image', img)

        # display after defined set of steps 
        if (count%steps_of) == 0 and count != amount_pictures:
            current_step = int(count / steps_of)
            print("\n [INFO] Next step commencing... \n")
            # only include when instructions are wanted
         #    print(user_instructions[current_step])
            # attention sound
            if PLAYSOUND: playsound('doubleTap.wav')
            print(str(count))
            input("Reposition your head and press <return> to continue.")


    k = cv2.waitKey(100) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break
    elif count >= amount_pictures: # Stop loop when the amount of pictures is collected
         break


if PLAYSOUND: playsound('doubleTap.wav')
# Do a bit of cleanup
print("\n [INFO] Exiting Program.")
cam.release()
cv2.destroyAllWindows()
