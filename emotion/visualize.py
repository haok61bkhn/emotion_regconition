
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.autograd import Variable
import emotion.transforms as transforms
from skimage import io
from skimage.transform import resize
from emotion.models import *
class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
cut_size = 44
net = None
checkpoint=None
transform_test=None
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
def predict(image):
    global class_names,net,checkpoint
    raw_img = image
    gray = rgb2gray(raw_img)
    gray = resize(gray, (48,48), mode='symmetric').astype(np.uint8)
    img = gray[:, :, np.newaxis]
    img = np.concatenate((img, img, img), axis=2)
    img = Image.fromarray(img)
    inputs = transform_test(img)
    ncrops, c, h, w = np.shape(inputs)
    inputs = inputs.view(-1, c, h, w)
    inputs = inputs.cuda()
    # inputs = Variable(inputs, volatile=True)
    outputs = net(inputs)
    outputs_avg = outputs.view(ncrops, -1).mean(0)  # avg over crops
    score = F.softmax(outputs_avg,dim=0)
    _, predicted = torch.max(outputs_avg.data, 0)
    return class_names[int(predicted.cpu().numpy())],score.data.cpu().numpy()[int(predicted.cpu().numpy())]

def Init():
    global transform_test,net,checkpoint
    transform_test = transforms.Compose([
    transforms.TenCrop(cut_size),
    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
    ])
    net = VGG('VGG19')
    checkpoint = torch.load(os.path.join('emotion/FER2013_VGG19', 'PrivateTest_model.t7'))
    net.load_state_dict(checkpoint['net'])
    net.cuda()
    net.eval()
# if __name__ == "__main__":
#     Init()
#     print(predict(io.imread('images/faker.jpg')))
#     print(predict(io.imread('images/hao.jpg')))
