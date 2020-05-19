import cv2
import os
from os.path import join, dirname, exists

#########################################################################
# SET PARAMETERS
#########################################################################

# set amount of pictures and pictures per step setting
PICTURES  = 80
STEP = 20

# additional space around the ear to be captured
# 0.1 is tightly around, 0.2 more generous 
SCALING = 0.2

INSTRUCTIONS = ["\n [INFO] Initializing ear capture. Turn your head left. Your right ear should then be facing the camera.", 
                "Look into the camera and slowly turn your head 45 degrees to the left",
                "Now look up, keeping the right ear towards the camera.",
                "Now look down, keeping the right ear towards the camera."
                ]
#########################################################################
# assert PICTURES/10 <= (len(user_instructions))

def capture_ear_images(amount_pic=PICTURES, pic_per_stage=STEP, margin=SCALING, is_authentification=False):

    cam = cv2.VideoCapture(0)
    # open window dimensions
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640) # set Width
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480) # set Height

    ear_detector = cv2.CascadeClassifier('Cascades/haarcascade_mcs_rightear.xml')

    # For each person, enter a new identification name
    ear_name = input('\n Enter name end press <return> ==>  ')

    dataset_dir = join(dirname(os.getcwd()), 'dataset')
    if not exists(dataset_dir):
        os.mkdir(dataset_dir)

    usr_dir = (join(dataset_dir, ear_name), join(dataset_dir, (ear_name + '-auth')))[is_authentification]
    if not exists(usr_dir):
        os.mkdir(usr_dir)

    print(INSTRUCTIONS[0])

        
    # Initialize individual sampling ear count
    count = 0

    while True:
        # receive image
        ret, frame = cam.read()
        # flip video frame horizontally to show it "mirror-like"
        frame = cv2.flip(frame, 1)
        rects = ear_detector.detectMultiScale(frame, 1.1, 5)

        for (x,y,w,h) in rects:
            # bounding box will be bigger by increasing the scaling
            left = x - int(w * margin)
            top = y - int(h * margin)
            right = x + int(w * (1+margin))
            bottom = y + int(h * (1+margin))
            green = (0,255,0)
            cv2.rectangle(frame, (left, top), (right, bottom), color=green, thickness=1)   
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
                # only include when instructions are wanted
    #             current_step = int(count / pic_per_stage)
    #             print(INSTRUCTIONS[current_step])
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
    cam.release()
    cv2.destroyAllWindows()
    

if __name__=='__main__':
    capture_ear_images()    
