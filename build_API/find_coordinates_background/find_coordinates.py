import cv2
coordinate=[]
image = cv2.imread("background.png")

# for i in range(1,image.shape[0]-1):
#    for j in range(1,image.shape[1]-1):
#        if (image[i][j][0] == 255 and image[i][j][1] == 255 and image[i][j][2] == 255 and image[i+1][j][0] != 255 and image[i][j+1][0] != 255):
#           coordinate.append({i, j})
#           print(str(i)+" "+str(j))
# for y in coordinate:
# 21 26
# 21 1604
# 143 612
# 300 25
# 309 1604
# 581 29
# 592 1601

# 203 200
# 203 1778
# 482 199
# 491 1778
# 559 1161
# 763 203
# 774 1775
coors = [[21, 26], [21, 1604], [300, 25], [309, 1604], [581, 29], [592, 1601], [143, 612]]
coorsend=[[203,200],[203,1778],[482,199],[491,1778],[763,203],[774,1775],[559,1161]]
for i in range(4,5):
  for x in range(coors[i][0], coorsend[i][0] + 1):
    for x1 in range(coors[i][1],coorsend[i][1]+1):
      image[x][x1][0] = 100
      image[x][x1][1] = 100
      image[x][x1][2] = 100
cv2.imwrite("image1.jpg",image)
