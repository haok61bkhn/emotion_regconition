# coding=utf-8
import mxnet as mx
import numpy as np
from Arcface import ArcfaceModel
from face_preprocess import preprocess
import cv2
from Sshdetector import SSHDetector
from OnetLnet import OnetLnetAlignment
import time

class FacialRecognition():
    def __init__(self, gpu_index=0, mtcnn_model="mtcnn-model", arcface_model="model-r100-ii/model,0",
                 image_size='112,112', ssh_detector="ssh-model-final/sshb", mtcnn_num_worker=2):
        if gpu_index >= 0:
            mtcnn_ctx = mx.gpu(gpu_index)
        else:
            mtcnn_ctx = mx.cpu()
        self.face_detector = SSHDetector(prefix=ssh_detector, epoch=0, ctx_id=gpu_index, test_mode=True)
        self.face_recognition = ArcfaceModel(gpu=gpu_index, model=arcface_model, image_size=image_size)
        self.landmark_detector = OnetLnetAlignment(model_folder=mtcnn_model, ctx=mtcnn_ctx, num_worker=mtcnn_num_worker,
                                                   accurate_landmark=True, threshold=[0.6, 0.7, 0.5])

    def get_scales(self, img):
        TEST_SCALES = [100, 200, 300, 400]
        target_size = 400
        max_size = 1200
        im_shape = img.shape
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])
        im_scale = float(target_size) / float(im_size_min)
        # prevent bigger axis from being more than max_size:
        if np.round(im_scale * im_size_max) > max_size:
            im_scale = float(max_size) / float(im_size_max)
        scales = [float(scale) / target_size * im_scale for scale in TEST_SCALES]
        return scales

    def detect_face_and_get_embedding(self, img):
        thresh = 0.95
        scales = self.get_scales(img)
        begin=time.time()
        bboxes = self.face_detector.detect(img, threshold=thresh, scales=scales)
        print("detection time : "+str(time.time()-begin))
        if len(bboxes) <= 0:
            return None
        rs = self.landmark_detector.detect_landmark(img, bboxes)
        if rs is not None:
            _, points = rs
            embeddings=[]
            for x in range(len(points)):
                point = points[x,:].reshape((2, 5)).T
                nimg = preprocess(img, bboxes[x], point, image_size='112,112')
                nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
                x1= np.transpose(nimg, (2, 0, 1))
                embeddings.append(self.face_recognition.get_feature(x1))
            return embeddings,bboxes,point
        return None

    def get_embedding(self, img):
        nimg = cv2.resize(img, (112, 112))
        nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
        x = np.transpose(nimg, (2, 0, 1))
        embeddings = self.face_recognition.get_feature(x)
        return embeddings
