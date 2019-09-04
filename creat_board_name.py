import cv2
import numpy as np

a=np.zeros((30,174,3))
for i in range(30):
  for j in range(174):
    # R: 124 G: 250 B: 251
    a[i][j][1] = 139
    a[i][j][2] = 0
    a[i][j][0] = 22
# font = cv2.FONT_HERSHEY_SIMPLEX
# cv2.putText(a,'Hao : Sad',(10,25), font, 0.7,(0,0,255),2,cv2.LINE_AA)
cv2.imwrite("boardname.jpg", a)
