import os
import cv2
from base_camera import BaseCamera
from process_Image import *

class Camera(BaseCamera):
    video_source = 0
    
    def __init__(self):
        self.predict=process_Image()
        if os.environ.get('OPENCV_CAMERA_SOURCE'):
            Camera.set_video_source(int(os.environ['OPENCV_CAMERA_SOURCE']))
        super(Camera, self).__init__()

    @staticmethod
    def set_video_source(source):
        Camera.video_source = source

    @staticmethod
    def frames(self):
        camera = cv2.VideoCapture(Camera.video_source)
        if not camera.isOpened():
            raise RuntimeError('Could not start camera.')

        while True:
            # read current frame
            _, img = camera.read()
            self.predict.setImage(img)
            self.predict.recognition()
            img=self.predict.image
            # img=cv2.resize(img,(470,615))
            # encode as a jpeg image and return it
            yield cv2.imencode('.jpg', img)[1].tobytes()
