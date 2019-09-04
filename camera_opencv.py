import os
import cv2
from base_camera import BaseCamera
from process_Image import *
import time
def over_draw(stt, img, image1, emotion):
      
        coors = [[21, 26], [21, 1604], [300, 25], [309, 1604], [581, 29], [592, 1601], [143, 612]]
        coorsend=[[203,200],[203,1778],[482,199],[491,1778],[763,203],[774,1775],[559,1161]]
        x = coors[stt][0]
        y = coors[stt][1]
        x1 = coorsend[stt][0]
        y1 = coorsend[stt][1]

        size = ( y1 - y+1,x1 - x+1)
        img = cv2.resize(img, size)
        image1[x:x1 + 1, y:y1 + 1,:] = img
        if (stt != 6):
            imgboard=cv2.imread("boardname.jpg")
            cv2.putText(imgboard,emotion, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
            image1[x1+5:35+x1,y:174+y,:]=imgboard
    
        return image1
class Camera(BaseCamera):
    video_source = 0
    
    def __init__(self):
        if os.environ.get('OPENCV_CAMERA_SOURCE'):
            Camera.set_video_source(int(os.environ['OPENCV_CAMERA_SOURCE']))
        super(Camera, self).__init__()
    
    
    @staticmethod
    def set_video_source(source):
        Camera.video_source = source

    @staticmethod
    def frames():
        d = 1
        d1=0
        image=cv2.imread("background.png")
        camera = cv2.VideoCapture(Camera.video_source)
        # camera=cv2.VideoCapture("test.mp4")
        if not camera.isOpened():
            raise RuntimeError('Could not start camera.')
        predict=Process()
        while True:
            # read current frame
            begin = time.time()
            _, img = camera.read()
            predict.setImage(img)
            predict.recogniton()
            img=predict.image
            # encode as a jpeg image and return it
            image = over_draw(6, img, image,None)
            if (predict.count >= d):
              for x in range(d,predict.count+1):
                    imgmini = cv2.imread("res_face/" + str(x-1) + ".jpg")
                    d += 1
                    over_draw((x-1) % 6 , imgmini, image,predict.name[x-1]+" : " +predict.enotion) 
            print("time : " + str(time.time() - begin))
            cv2.imwrite("res/" + str(d1) + ".jpg", image)
            d1+=1
            cv2.waitKey(20)
            yield cv2.imencode('.jpg', image)[1].tobytes()
