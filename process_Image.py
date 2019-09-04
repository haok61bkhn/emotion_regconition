from ArcfaceSshOLnet import *
import cv2
import numpy as np
from scipy.spatial import distance
import pickle
import time
import glob
import turtle
import random
from webcolors import name_to_rgb
from emotion.visualize import *
from skimage import io
from matplotlib import pyplot as plt
import cv2

class Process:
  def __init__(self):
    Init()
    self.transform_test = transforms.Compose([
    transforms.TenCrop(cut_size),
    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
    ])
    self.net = VGG('VGG19')
    self.checkpoint = torch.load(os.path.join('emotion/FER2013_VGG19', 'PrivateTest_model.t7'))
    self.net.load_state_dict(self.checkpoint['net'])
    self.net.cuda()
    self.net.eval()
    f=open("train_sort_v100_ssh_v2.pkl","rb")
    self.model = pickle.load(f)
    self.arcf = FacialRecognition(gpu_index=-1)
    self.image = None
    self.count=0
    self.enotion=None
    self.name=[]
  def setImage(self,img):
    self.image=img
  
  def random_color(self):
    colors = ["red", "green", "blue", "orange", "purple", "pink", "yellow"]
    color = random.choice(colors)
    return name_to_rgb(color)


  def draw(self,box,text):
    color = self.random_color()
    img, point = preprocess(self.image, box, None)  # y1=>y2 x1=>x2
    eimage = self.image[point[0]:point[1], point[2]:point[3]]
    self.count += 1
    cv2.imwrite("res_face/"+str(self.count-1)+".jpg",eimage)
    emotion = predict(eimage)
    self.name.append(text)
    self.enotion=emotion[0]
    cv2.rectangle(self.image, (point[3], point[1]), (point[2], point[0]),color, 3)
    cv2.putText(self.image, text, (point[2], point[0] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
    print( text+" "+emotion[0]+" "+str(emotion[1]))
  def recogniton(self):
    try:
      b1, boxx1, p = self.arcf.detect_face_and_get_embedding(self.image)
      for i in range(len(b1)):
        b = b1[i]
        boxx = boxx1[len(b1) - i - 1]
        dist = 100
        maxdist=0
        res = ""
        for x in self.model:
          a = x['features']
          try:
            if (distance.euclidean(a, b) < dist): res = x['class']
            maxdist=max(dist,distance.euclidean(a, b))
            dist = min(dist, distance.euclidean(a, b))
          except:
            x = True
        print("dist = "+str(dist))
        if dist>1.23: res="Nguoi la"
        self.draw(boxx, res)
    except:
        print("false")
# if __name__ == "__main__":
#   x = Process()
#   x.setImage(cv2.imread("/home/hbhb/home_ubuntu/lab_Thầy_Thuận/face_recoginition/AIVIVN2/solution/test/songtung1.jpg"))
#   x.recogniton()
#   cv2.imshow("image",x.image)
#   cv2.waitKey(0)

#   x.setImage(cv2.imread("/home/hbhb/home_ubuntu/lab_Thầy_Thuận/face_recoginition/AIVIVN2/solution/test/hao1.jpg"))
#   x.recogniton()
#   cv2.imshow("image",x.image)
#   cv2.waitKey(0)
