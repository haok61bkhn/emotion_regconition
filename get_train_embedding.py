import glob
import pickle
from  ArcfaceSshOLnet import FacialRecognition
import cv2
import argparse
import time
from mydata.create_sorted import *
class Traindata:
    def __init__(self):
        self.mtcnn_model='mtcnn-model'     
        self.arcface_model='model-r100-ii/model,0'
        self.ssh_detector='ssh-model-final/sshb'
        self.file_output="train_sort_v100_ssh_v2.pkl"
        self.flip='0'
        self.image_folder='mydata/'
        self.image_file='mydata/sorted-train.txt'
        self.gpu_index=-1
        self.flip = False
        self.model = FacialRecognition(mtcnn_model=self.mtcnn_model, arcface_model=self.arcface_model,
                              ssh_detector=self.ssh_detector, gpu_index=self.gpu_index, mtcnn_num_worker=4)

    def convert_str2bool(arg_str, default_v):
        if isinstance(arg_str, str):
            if arg_str.lower() in {"1", "y", "yes"}:
                return True
            if arg_str.lower() in {"0", "n", "no"}:
                return False
        if isinstance(arg_str, bool):
            return arg_str
        return default_v
    def train(self):
        create_sortedtrain()
        dicts=[]
        try:
            f = open("train_sort_v100_ssh_v2.pkl","rb")
            dicts = pickle.load(f)
            f.close()
        except:
            dicts=[]
        wr = open(self.file_output, "wb")
        f = open(self.image_file, "r")
        error = 0
        for line in f:
            dict = {}
            line = line.replace("\n", "")
            label = line.split()[0]
            img_file = line.split()[1]
            print(img_file)
            begin = time.time()
            img = cv2.imread(self.image_folder + img_file)
            embedding=None
            try:
                embedding, k,p = self.model.detect_face_and_get_embedding(img)
            except:
                print("")
            if embedding is None:
                error += 1
                print("false : "+img_file)
            if embedding is not None:
                dict['class'] = label
                dict['features'] = embedding
                dict['imgfile'] = img_file
                dicts.append(dict)
            if self.flip:
                img = cv2.flip(img, 1)
                embedding,k,p = self.model.detect_face_and_get_embedding(img)
                if embedding is not None:
                    dict['class'] = label
                    dict['features'] = embedding
                    dict['imgfile'] = img_file
                    dicts.append(dict)
            print("time : "+str(time.time()-begin))
        print ("So buc anh ko detect duoc face ", error)
        pickle.dump(dicts, wr)
        wr.close()

if __name__ == "__main__":
    tdt = Traindata()
    tdt.train()
