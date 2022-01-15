import numpy as np
import cv2 as cv
import copy
import time

#Tuning parameters of your choice
MHI_DURATION = 3
thrs=50
count = 0 

# (empty) trackbar callback
def nothing( ):
    pass

if __name__ == '__main__':
    import sys
    try:
        video_src = sys.argv[1]
    except:
        video_src = 0

    cam = cv.VideoCapture(video_src)
    # Set window size : 320x240 pixels
    cam.set(3,320) 
    cam.set(4,240) 
    ret, frame = cam.read()
    frame=cv.flip(frame,1)
    h, w = frame.shape[:2]
    prev_frame = frame.copy()
    motion_history = np.zeros((h, w), np.float32)
    mhi = np.zeros((h, w), np.float32)
    while True:
        ret, frame = cam.read()
        frame=cv.flip(frame,1)
        if ret == False:
            break
        frame_diff = cv.absdiff(frame, prev_frame)
        gray_diff = cv.cvtColor(frame_diff, cv.COLOR_BGR2GRAY)
        
        ret, motion_mask = cv.threshold(gray_diff, thrs, 1, cv.THRESH_BINARY)
        
        
        timestamp = cv.getTickCount() / cv.getTickFrequency()
        cv.motempl.updateMotionHistory(motion_mask, motion_history, timestamp, MHI_DURATION)
        vis = np.uint8(np.clip((motion_history-(timestamp-MHI_DURATION)) / MHI_DURATION, 0, 1)*255)
        vis = cv.cvtColor(vis, cv.COLOR_GRAY2BGR)
        frame1 = copy.deepcopy(frame)

        cv.line(frame1, (80,0), (80,240), (255,0,0), 2)
        cv.line(frame1, (160,0), (160,240), (255,0,0), 2)
        cv.line(frame1, (240,0), (240,240), (255,0,0), 2)

        cv.line(frame1, (0,60), (320,60), (255,0,0), 2)
        cv.line(frame1, (0,120), (320,120), (255,0,0), 2)
        cv.line(frame1, (0,180), (320,180), (255,0,0), 2)
        cv.imshow('Motion_history_image',cv.resize(vis, dsize=None, fx=1, fy=1))
        #cv.imshow('Mesh',cv.resize(frame1, dsize=None, fx=1, fy=1))
        cv.imshow('Original',cv.resize(frame, dsize=None, fx=1, fy=1))

        prev_frame = frame.copy()
        if 0xFF & cv.waitKey(5) == 27:
            #Press ESC to exit window
            break
        elif cv.waitKey(5) ==32:
        #Press space to take picture
            print("image_"+str(count)+"saved")
            #cv.imwrite(<your PATH> +str(count)+'.jpg', vis)
            count +=1
                        
cam.release ()
cv.destroyAllWindows ()
