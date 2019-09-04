import cv2
import time

coors = [[21, 26], [21, 1604], [300, 25], [309, 1604], [581, 29], [592, 1601], [143, 612]]
coorsend=[[203,200],[203,1778],[482,199],[491,1778],[763,203],[774,1775],[559,1161]]
def over_draw(stt,img):
  global image1
  
  x = coors[stt][0]
  y = coors[stt][1]
  x1 = coorsend[stt][0]
  y1 = coorsend[stt][1]

  size = ( y1 - y+1,x1 - x+1)
  img = cv2.resize(img, size)
  image1[x:x1+1,y:y1+1,:]=img
  # for i in range(x, x1 + 1):
  #   for j in range(y, y1 + 1):
  #      for k in range(0, 3):
  #        image1[i][j][k]=img[i-x][j-y][k]
  
if __name__ == "__main__":
  image1 = cv2.imread("background.png")
 
  camera = cv2.VideoCapture(0)
  while True:
        grab, frame = camera.read()
        image = cv2.resize(frame, ( 550,417))
        begin=time.time()
        over_draw(6, image)
        over_draw(0, image)
        over_draw(1, image)
        over_draw(2, image)
        over_draw(3, image)
        over_draw(4, image)
        over_draw(5, image)


        print(time.time()-begin)
        cv2.imshow("detection result", image1)
        cv2.waitKey(25)
