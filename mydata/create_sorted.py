import os
import glob
import pickle
def create_sortedtrain():
  try:
   f = open("train_sort_v100_ssh_v2.pkl","rb")
   model = pickle.load(f)
  
  except:
    model=[]
  labels = []
  for y in model:
    x=y['class']
    if x not in labels:
      labels.append(x)
  print(labels)
  try:
    os.remove("mydata/sorted-train.txt")
  except:
    print("begin")
  f=open("mydata/sorted-train.txt","w+")
  fordername = glob.glob("mydata/*")
  delfordername = glob.glob("mydata/*.py")
  for y in fordername:
    x = y[7:]
    if x not in delfordername and x not in labels:
      listname = glob.glob("mydata/"+x + "/*.jpg")
      for label in listname:
        f.write(x+ " "  +label[7:]+ "\n")
