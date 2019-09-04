from emotion.visualize import *
from skimage import io
from matplotlib import pyplot as plt
import cv2
if __name__ == "__main__":
    Init()
    image1=cv2.imread('emotion/images/faker.jpg')
    image2=io.imread('emotion/images/faker.jpg')
    #io.imshow(image[:,:,3])
    #plt.show()
    print(predict(image1))
    print(predict(image2))
