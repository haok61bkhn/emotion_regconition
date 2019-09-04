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
image = None
d = 0
arcf=None
def random_color():
  colors = ["red", "green", "blue", "orange", "purple", "pink", "yellow"]
  color = random.choice(colors)
  return name_to_rgb(color)
def draw(box,text):
  global image
  color = random_color()
  img, point = preprocess(image, box, None)  # y1=>y2 x1=>x2

  eimage = image[point[0]:point[1], point[2]:point[3]]
  emotion = predict(eimage)
  cv2.rectangle(image, (point[3], point[1]), (point[2], point[0]),color, 3)
  cv2.putText(image, text, (point[2], point[0] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
  print( text+" "+emotion[0]+" "+str(emotion[1]))
def recogniton(path):
 global arcf, image, d
 try:

  # image=cv2.imread(path)
  b1, boxx1,p = arcf.detect_face_and_get_embedding(image)
  # for point in p:
  #   x = point[0]
  #   y = point[1]
  #   print(y)
  #   cv2.circle(img,(x,y), 2, (0,0,255), -1)
  
  for i in range(len(b1)):
    b = b1[i]
    boxx=boxx1[len(b1)-i-1]  
# image = arcf.get_scales(img)
# cv2.imshow("image", img)
# cv2.waitKey(0)
    dist = 100
    maxdist=0
    res = ""
    for x in model:
      a = x['features']
      try:
          if (distance.euclidean(a, b) < dist): res = x['class']
          maxdist=max(dist,distance.euclidean(a, b))
          dist = min(dist, distance.euclidean(a, b))
      except:
           x = True
    print("dist = "+str(dist))
    if dist>1.10: res="Nguoi la"
    draw(boxx, res)
 except:
    print("false")
  
  # cv2.imshow("image", image)
  # cv2.waitKey(0)
  # cv2.imwrite("res/Chi_Pu/"+str(d) + ".jpg", image)
  # d += 1
#  except:
#    print("false")

def Camera():
  global image
  camera = cv2.VideoCapture(0)
  begin = time.time()
  while True:
        begin = time.time()
        grab, frame = camera.read()
        image = cv2.resize(frame, (500, 500))
        recogniton("none")
        cv2.imshow("detection result", image)
        print("time : " + str(time.time()-begin))
        cv2.waitKey(10)

def testImage():
  global image
  while (True):
  #  try:
    print("Path : ")
    path = input()
    begin = time.time()
    image = cv2.imread(path)
    recogniton("none")
    cv2.imshow("image", image)
    print("total time : " + str(time.time()-begin))
    cv2.waitKey(0)
  #  except:
  #   print("path false")
  #   break
if __name__ == "__main__":
  Init()
  f=open("train_sort_v100_ssh_v2.pkl","rb")
  model = pickle.load(f)
  arcf = FacialRecognition(gpu_index=-1)
  # testImage()
  Camera()
