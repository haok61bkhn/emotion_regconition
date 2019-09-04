from Sshdetector import *
import cv2
from face_preprocess import *
from OnetLnet import *
import time
import pickle
image= None
def draw(det1):
  global image
  for det in det1:
    img,point=preprocess(image,det,None) # y1=>y2 x1=>x2
    cv2.rectangle(image, (point[3], point[1]), (point[2], point[0]), (200, 200, 100), 3)
    
    

if __name__ == "__main__":
  # image=cv2.imread("t1.jpg")
  # detect = SSHDetector("ssh-model-final/sshb", 0)
  # det1 = detect.detect(image)
  # draw([det1[3]])
  # cv2.imshow("image", image)
  # cv2.waitKey(0)
