import numpy as np
import cv2 as cv
import math
import os
import time

from numpy.core.fromnumeric import cumprod
# print( cv.__version__)

# tester = np.ones([2,3])
# print(tester)
# print(tester.shape[1])

# dirname = os.path.dirname(__file__)
# imagLpath = dirname + "\\MiddEval3-data-Q\\MiddEval3\\trainingQ\\Motorcycle\\im0.png"

# imgL = cv.imread(cv.samples.findFile(imagLpath))
# cv.imshow("Left", imgL)
# k = cv.waitKey(0)

# for i in range(0,10):
#     # cv.rectangle(imgL, (20+40*i,20),(40+40*i,40),(0,255,0),3)
#     imgL[20:40,(20+40*i):(40+40*i),:] = 255
#     cv.imshow("Left", imgL)
#     k = cv.waitKey(0)


# tester = np.array([[1,1],[1,1]])
# tester2 = np.array([[1,1],[1,1]])

# N = np.sum(abs(abs(tester) - abs(tester2)))

# print(N)

curPoint = np.array((1,1))
corresPoint = np.array((2,2))


t = time.time()
dist = abs(np.linalg.norm(curPoint - corresPoint))
elapsed = time.time()-t

t = time.time()
dist2 = np.sqrt((curPoint[0]-corresPoint[0])**2 + (curPoint[1]-corresPoint[1])**2)
elapsed2 = time.time()-t

print(dist)
print(dist2)
print(elapsed)
print(elapsed2)

# def selectCurWindow(image, windowSize, X, Y):
#             numPixRow = image.shape[1]
#             numPixCol = image.shape[0]

#             if ((X >= numPixRow) or (X < 0)) or ((Y >= numPixCol) or (Y < 0)):
#                 return -1

#             window = np.zeros([windowSize,windowSize])
#             offset = math.floor(windowSize/2)

#             for i in range(-offset,offset+1):
#                 for j in range(-offset,offset+1):
#                     if (((X + i) < 0) or ((X + i) >= numPixRow)):
#                         pass
#                     elif (((Y + j) < 0) or ((Y + j) >= numPixCol)):
#                         pass
#                     else:
#                         window[i+offset,j+offset]  = image[X+i,Y+j]

#             print(window,'\n')


            

# # tester = np.ones([10,10])
tester = np.array([[0,1,2,3,4,5,6,7,8,9],[1,2,3,4,5,6,7,8,9,0],[2,3,4,5,6,7,8,9,0,1],[3,4,5,6,7,8,9,0,1,2],[4,5,6,7,8,9,0,1,2,3],[5,6,7,8,9,0,1,2,3,4],[6,7,8,9,0,1,2,3,4,5],[7,8,9,0,1,2,3,4,5,6],[8,9,0,1,2,3,4,5,6,7],[9,0,1,2,3,4,5,6,7,8]])

print(np.max(tester))
print(255*tester/9)

# # selectCurWindow(tester, 5, 0,0)
# # selectCurWindow(tester, 5, 9,9)
# # selectCurWindow(tester, 5, 9,0)
# # selectCurWindow(tester, 5, 0,9)
# # selectCurWindow(tester, 5, 5,5)
# # selectCurWindow(tester, 5, 4,4)

# # selectCurWindow(tester, 5, 0,0)
# # selectCurWindow(tester, 5, 0,1)
# # selectCurWindow(tester, 5, 0,2)
# # selectCurWindow(tester, 5, 0,3)
# # selectCurWindow(tester, 5, 0,4)
# # selectCurWindow(tester, 5, 0,5)
# # selectCurWindow(tester, 5, 0,6)
# # selectCurWindow(tester, 5, 0,7)
# # selectCurWindow(tester, 5, 0,8)
# # selectCurWindow(tester, 5, 0,9)

# selectCurWindow(tester, 5, 0,0)
# selectCurWindow(tester, 5, 1,0)
# selectCurWindow(tester, 5, 2,0)
# selectCurWindow(tester, 5, 3,0)
# selectCurWindow(tester, 5, 4,0)
# selectCurWindow(tester, 5, 5,0)
# selectCurWindow(tester, 5, 6,0)
# selectCurWindow(tester, 5, 7,0)
# selectCurWindow(tester, 5, 8,0)
# selectCurWindow(tester, 5, 9,0)

            # if ((X - math.floor(windowSize/2)) < 0):
            #     offsetterX = -abs((X - math.floor(windowSize/2)))
            # elif ((X + math.floor(windowSize/2)) > numPixRow):
            #     offsetterX = abs((numPixRow - math.floor(windowSize/2)))

            # if ((Y - math.floor(windowSize/2)) < 0):
            #     offsetterY = -abs((Y - math.floor(windowSize/2)))
            # elif ((Y + math.floor(windowSize/2)) > numPixCol):
            #     offsetterY = abs((numPixCol - math.floor(windowSize/2)))

            # print([offsetterX, offsetterY])