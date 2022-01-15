import numpy as np
import cv2 as cv
from keras.models import load_model
import copy
import time

MHI_DURATION =  5
MAX_TIME_DELTA = 0.25
MIN_TIME_DELTA = 0.05
thrs=50
count = 0


gesture_names = {0:'uong',
                1: 'doi',
                2: 'vui',
                3: 'gian',
                4: 'toi',
                5:'none'
                }
        

# Load model từ file H5
model = load_model(<Your PATH>)

# Hàm predict 
def predict_rgb_image_vgg(image):
    #đọc ảnh dưới dạng array
    image = np.array(image, dtype='float32')
    
    image /= 255
    #tạo list tên pre_array, gia trị của list là kết quả sau khi đưa hình (img) vào model.
    pred_array = model.predict(image)
    print(f'pred_array: {pred_array}')
    #lấy ra cái index có giá trị value lón nhất trong pred_array,
    #dùng index đó làm value để lấy chữ ra trong dict gesture_name.
    result = gesture_names[np.argmax(pred_array)]
    #xét các key có giá trị lớn nhất theo chiều dọc từng chiều trong dic
    #result = np.max(result)
    print(f'Result: {result}')
    print(max(pred_array[0]))
    score = float("%0.2f" % (max(pred_array[0]) * 100))
    print(result)
    print(score)
    return result, score

def nothing( ):
    pass

if __name__=='__main__' :
    import sys
    
    try:
        video_src = sys.argv[1]
    except:
        video_src = 0
   
    cam = cv.VideoCapture(0)
    # cài đặt chiều ngang khung hình là 320 pixel
    cam.set(3,320) 
    #cài đặt chiều cao khung hình là 240 pixel
    cam.set(4,240) 
    ret, frame = cam.read()
    #frame=cv.flip(frame,0)
    frame=cv.flip(frame,1)
    h, w = frame.shape[:2]
    prev_frame = frame.copy()
    motion_history = np.zeros((h, w), np.float32)
    mhi = np.zeros((h, w), np.float32)
    while True:
        ret, frame = cam.read()
        #frame=cv.flip(frame,0)
        frame=cv.flip(frame,1)
        #cam.set(cv.CAP_PROP_FPS,30)
        fps = cam.get(cv.CAP_PROP_FPS)
        fps=str(fps)
        if ret == False:
            break
        frame_diff = cv.absdiff(frame, prev_frame)
        gray_diff = cv.cvtColor(frame_diff, cv.COLOR_BGR2GRAY)
        ret, motion_mask = cv.threshold(gray_diff, thrs, 1, cv.THRESH_BINARY)   
        timestamp = cv.getTickCount() / cv.getTickFrequency()
        cv.motempl.updateMotionHistory(motion_mask, motion_history, timestamp, MHI_DURATION)
      

        vis = np.uint8(np.clip((motion_history-(timestamp-MHI_DURATION)) / MHI_DURATION, 0, 1)*255)
        vis = cv.cvtColor(vis, cv.COLOR_GRAY2BGR)
        cv.imshow('Motion_history_image',cv.resize(vis, dsize=None, fx=1.5, fy=1.5))
    
        prev_frame = frame.copy()

        vis = cv.resize(vis, (160,160))
        vis = vis.reshape(-1,160,160, 3)
        result, score = predict_rgb_image_vgg(vis)
        frame1 = copy.deepcopy(frame)
        cv.line(frame1, (80,0), (80,240), (255,0,0), 2)
        cv.line(frame1, (160,0), (160,240), (255,0,0), 2)
        cv.line(frame1, (240,0), (240,240), (255,0,0), 2)
        cv.line(frame1, (0,60), (320,60), (255,0,0), 2)
        cv.line(frame1, (0,120), (320,120), (255,0,0), 2)
        cv.line(frame1, (0,180), (320,180), (255,0,0), 2)
        cv.putText(frame1, fps+" "+"fps",(12,50),cv.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2,lineType=cv.LINE_AA)
        
        cv.putText(frame1, str(count),(290,200),cv.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2,lineType=cv.LINE_AA)
        cv.putText(frame1, result ,(12,30),cv.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,lineType=cv.LINE_AA)
        cv.putText(frame1, "thrs" +" "+ str(thrs),(12,70),cv.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2,lineType=cv.LINE_AA)
        cv.imshow('Original',cv.resize(frame1, dsize=None, fx=1, fy=1))
               
        if 0xFF & cv.waitKey(5) == 27:
            #Nhấn ESC để tắt cửa sổ
            break
cv.destroyAllWindows()       
cam.release ()
