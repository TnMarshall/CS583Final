import numpy as np
import cv2 as cv
import math
import os
import copy
import time
from matplotlib import pyplot as plt
from numpy.core.numeric import correlate

class stereoSystem:

    def __init__(self, leftImName, rightImName, camLintr, camRintr, height, width, hzoffset, baseline):
        ''' Initialization function for the stereo system. Takes in
            leftImName:  file name of the left image
            rightImName: file name of the right image
            camLintr:    The camera matrix for the rectified view from the left camera.
            camRintr:    The camera matrix for the rectified view from the right camera.
            height:      The height of the pictures in pixels
            width:       The width of the pictures in pixels
            hzoffset:    The x-difference horizontal offset between the principal points of the cameras. 
            baseline:    camera baseline in mm'''

        self.imgL = cv.imread(cv.samples.findFile(leftImName))
        self.imgR = cv.imread(cv.samples.findFile(rightImName))
        self.Lintr = camLintr
        self.Rintr = camRintr
        self.height = height
        self.width = width
        self.hzOffset = hzoffset
        self.baseline = baseline
        self.C0 = np.array([self.Lintr[0,2],self.Lintr[1,2]])
        self.C1 = np.array([self.Rintr[0,2],self.Rintr[1,2]])
        self.f = self.Lintr[0,0]

        self.extrinsT = np.array([[baseline],[0],[0]])

        self.imgLg = cv.cvtColor(self.imgL, cv.COLOR_BGR2GRAY)
        self.imgRg = cv.cvtColor(self.imgR, cv.COLOR_BGR2GRAY)

        # WRONG maybe
        # # the fundamental matrix in this case is just extrinsT because the cameras are axis-aligned
        # self.fund = self.extrinsT

        # cv.imshow("Left", self.imgL)
        # cv.imshow("Right", self.imgR)
        # k = cv.waitKey(0)

    def calculateEpipolar(self, X, Y):
        '''Calculates the epipolar line in image 2 corresponding to the X,Y coordinate in image 1
            Returns: endpoints- an array with two points in it that constitute the endpoints of the epipolar line
                    in the second image.'''
        # Not needed if you can find it in open cv
        # if the images are axis aligned, then the epipolar line is just the corresponding horizontal array in the right image
        epiY = Y
        epiXL = 0
        epiXR = self.width-1
        endpoints = np.array([[epiXL,epiY],[epiXR,epiY]])
        return endpoints

    def drawEpipolar(self, X, Y):
        '''Inputs:
            X - the x coordinate of a point in the left image
            Y - the y coordinate of a point in the right image
            
            Returns:
            Nothing
            
            Action: Displays the left and right images with the original point highlighted in the left image and the 
            corresponding epipolar line displayed in the right image.'''
        # Not needed if you can find it in open CV
        imgL = copy.deepcopy(self.imgL)
        imgR = copy.deepcopy(self.imgR)

        endpoints = self.calculateEpipolar(X,Y)

        # imgL[X,Y,:] = [255,0,0] # not visible
        end1 = endpoints[0,:]
        end2 = endpoints[1,:]
        cv.line(imgR,end1,end2, (0,0,255), 1)
        cv.circle(imgL, (X,Y), 3, (255,0,0), 2)

        cv.imshow("Left", imgL)
        cv.imshow("Right", imgR)
        k = cv.waitKey(0)

    def findFundMatr(self):

        ###### FOR SELECTING POINTS BY HAND ######
        # leftPoints = []
        # rightPoints = []
        # # imgL = copy.deepcopy(self.imgL)
        # # imgR = copy.deepcopy(self.imgR)
        # imgL = cv.cvtColor(self.imgL, cv.COLOR_BGR2GRAY)
        # imgR = cv.cvtColor(self.imgR, cv.COLOR_BGR2GRAY)

        # cv.imshow("Left", imgL)
        # cv.imshow("Right", imgR)

        # def collectPointandMarkL(event, x, y, flags, param):
        #     if event == cv.EVENT_LBUTTONDBLCLK:
        #         cv.circle(imgL, (x,y), 5, (255,0,0),3)
        #         leftPoints.append([x,y])
        # def collectPointandMarkR(event, x, y, flags, param):
        #     if event == cv.EVENT_LBUTTONDBLCLK:
        #         cv.circle(imgR, (x,y), 5, (255,0,0),3)
        #         rightPoints.append([x,y])
        
        # cv.setMouseCallback('Left', collectPointandMarkL)
        # cv.setMouseCallback('Right', collectPointandMarkR)

        # while(1):
        #     cv.imshow("Left", imgL)
        #     cv.imshow("Right", imgR)
        #     if cv.waitKey(20) & 0xFF == 102:  #27 is esc 102 is f
        #         break

        # # print(leftPoints)
        # # print(rightPoints)
        # cv.destroyAllWindows()
        

        # pts1 = np.int32(leftPoints)
        # pts2 = np.int32(rightPoints)
        # F, mask = cv.findFundamentalMat(pts1,pts2,cv.FM_LMEDS)

        
        # pts1 = pts1[mask.ravel()==1]
        # pts2 = pts2[mask.ravel()==1]

        # # Find epipolar lines
        # linesL = cv.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
        # linesL = linesL.reshape(-1,3)
        
        # linesR = cv.computeCorrespondEpilines(pts2.reshape(-1,1,2), 1,F)
        # linesR = linesR.reshape(-1,3)

        # # Draw epipolar lines

        # def drawEpiLines(imgL, imgR, lines, pts1, pts2):
        #     r,c = imgL.shape
        
        #     for r,pts1,pts2 in zip(lines, pts1, pts2):
        #         lineColor = tuple(np.random.randint(0,255,3).tolist())
        #         x0,y0 = map(int, [0, -r[2]/r[1]])
        #         x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1]])
        #         imgL = cv.line(imgL, (x0, y0), (x1,y1), lineColor, 1)
        #         imgL = cv.circle(imgL, tuple(pts1),5,lineColor,-1)
        #         imgR = cv.circle(imgR, tuple(pts2),5,lineColor,-1)
            
        #     return imgL, imgR
        
        # imgLout, imgRnan = drawEpiLines(imgL, imgR, linesL, pts1, pts2)
        # imgRout, imgLnan = drawEpiLines(imgR, imgL, linesR, pts2, pts1)
        ###### FOR SELECTING POINTS BY HAND ######


        ##### Modified FROM OPENCV TUTORIAL https://docs.opencv.org/master/da/de9/tutorial_py_epipolar_geometry.html #####

        # img1 = cv.cvtColor(self.imgL, cv.COLOR_BGR2GRAY)
        # img2 = cv.cvtColor(self.imgR, cv.COLOR_BGR2GRAY)

        img1 = self.imgLg
        img2 = self.imgRg

        sift = cv.SIFT_create()
        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(img1,None)
        kp2, des2 = sift.detectAndCompute(img2,None)
        # FLANN parameters
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50)
        flann = cv.FlannBasedMatcher(index_params,search_params)
        matches = flann.knnMatch(des1,des2,k=2)
        pts1 = []
        pts2 = []
        # ratio test as per Lowe's paper
        for i,(m,n) in enumerate(matches):
            if m.distance < 0.8*n.distance:
                pts2.append(kp2[m.trainIdx].pt)
                pts1.append(kp1[m.queryIdx].pt)

        pts1 = np.int32(pts1)
        pts2 = np.int32(pts2)
        F, mask = cv.findFundamentalMat(pts1,pts2,cv.FM_LMEDS)
        self.fundMatr = F
        # We select only inlier points
        pts1 = pts1[mask.ravel()==1]
        pts2 = pts2[mask.ravel()==1]

        def drawlines(img1,img2,lines,pts1,pts2):
            ''' img1 - image on which we draw the epilines for the points in img2
                lines - corresponding epilines '''
            r,c = img1.shape
            img1 = cv.cvtColor(img1,cv.COLOR_GRAY2BGR)
            img2 = cv.cvtColor(img2,cv.COLOR_GRAY2BGR)
            for r,pt1,pt2 in zip(lines,pts1,pts2):
                color = tuple(np.random.randint(0,255,3).tolist())
                x0,y0 = map(int, [0, -r[2]/r[1] ])
                x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
                img1 = cv.line(img1, (x0,y0), (x1,y1), color,1)
                img1 = cv.circle(img1,tuple(pt1),5,color,-1)
                img2 = cv.circle(img2,tuple(pt2),5,color,-1)
            return img1,img2

        # Find epilines corresponding to points in right image (second image) and
        # drawing its lines on left image
        lines1 = cv.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
        lines1 = lines1.reshape(-1,3)
        img5,img6 = drawlines(img1,img2,lines1,pts1,pts2)
        # Find epilines corresponding to points in left image (first image) and
        # drawing its lines on right image
        lines2 = cv.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
        lines2 = lines2.reshape(-1,3)
        img3,img4 = drawlines(img2,img1,lines2,pts2,pts1)
        # plt.subplot(121),plt.imshow(img5)
        # plt.subplot(122),plt.imshow(img3)
        # plt.show()

        ##### Modified FROM OPENCV TUTORIAL https://docs.opencv.org/master/da/de9/tutorial_py_epipolar_geometry.html #####
        
        cv.imshow("Left", img5)
        cv.imshow("Right", img3)
        k = cv.waitKey(0)


    def findCorrespondant(self, X,Y, windowSize):
        '''Because the images are rectified, the epipolar lines are horizontal
            X = x coord in left image
            Y = y coord in right image'''

        def selectCurWindow(image, windowSize, X, Y):
            # numPixRow = image.shape[1]
            # numPixCol = image.shape[0]

            # Switched due to numpy orientation
            numPixRow = image.shape[0]
            numPixCol = image.shape[1]


            if ((X >= numPixRow) or (X < 0)) or ((Y >= numPixCol) or (Y < 0)):
                return -1

            window = np.zeros([windowSize,windowSize])
            offset = math.floor(windowSize/2)

            for i in range(-offset,offset+1):
                for j in range(-offset,offset+1):
                    # if (((X + i) < 0) or ((X + i) >= numPixRow)):
                    #     pass
                    # elif (((Y + j) < 0) or ((Y + j) >= numPixCol)):
                    #     pass
                    # else:
                    try:
                        window[i+offset,j+offset]  = image[X+i,Y+j]
                    except:
                        pass

            # print(window,'\n')
            return window
        
        numPixRow = self.width

        # windowSize = 201

        # img1 = cv.cvtColor(self.imgL, cv.COLOR_BGR2GRAY)
        # img2 = cv.cvtColor(self.imgR, cv.COLOR_BGR2GRAY)

        # t = time.time()
                
        img1 = self.imgLg
        img2 = self.imgRg

        # elapsed1 = time.time() - t

        # t = time.time()
        curWind = selectCurWindow(img1, windowSize, Y, X)
        # elapsed2 = time.time() - t

        offset = 60
        xMin = X - offset
        xMax = X + offset
        if xMax > numPixRow:
            xMax = numPixRow
        if xMin < 0:
            xMin = 0

        energyDifs = np.zeros([1,xMax-xMin])
        # t = time.time()
        for i in range(xMin,xMax):
            curCompWind = selectCurWindow(img2, windowSize, Y, i)
            # used for checking
            # curCompWind = selectCurWindow(img1, windowSize, Y, i)

            # N = np.sum(np.multiply(curWind,curCompWind))/(np.sum(np.sqrt(np.multiply(curWind,curWind)))*np.sum(np.sqrt(np.multiply(curCompWind,curCompWind))))
            N = np.sum(abs(abs(curWind) - abs(curCompWind)))
            energyDifs[0,i-xMin] = N
        # elapsed3 = time.time() - t
        # print(energyDifs)

        curInd = 0
        # curVal = abs(energyDifs[0,0])
        curInd = np.argmin(energyDifs)+xMin

        # for i in range(0,numPixRow):
        #     curCheck = abs(energyDifs[0,i])
        #     if not(curCheck == 1) and (energyDifs[0,i] < curVal):
        #         curVal = abs(energyDifs[0,i])
        #         curInd = i
                
        # print(curInd)
        # curCompWind = selectCurWindow(img2, windowSize, Y, curInd)
        # Used for checking
        # curCompWind = selectCurWindow(img1, windowSize, Y, curInd)

        # Visualize for checking
        # cv.imshow("curWind", np.array(curWind,dtype=np.uint8))
        # cv.imshow("curCompWind", np.array(curCompWind,dtype=np.uint8))
        # k = cv.waitKey(0)

        # print(elapsed1)
        # print(elapsed2)
        # print(elapsed3)

        return np.array([curInd, Y])

    def displayCorrespondent(self, X,Y, windowSize):
        [Xr, Yr] = self.findCorrespondant(X,Y, windowSize)

        img1 = copy.deepcopy(self.imgL)
        img2 = copy.deepcopy(self.imgR)

        cv.circle(img1, (X,Y), windowSize, (0,0,255), 5)
        cv.circle(img2, (Xr,Yr), windowSize, (0,0,255), 5)
        left = "Left X: " + str(X) + " Y: " + str(Y)
        right = "Right X: " + str(Xr) + " Y: " + str(Yr)
        # cv.imshow(left, img1)
        # cv.imshow(right, img2)
        buffer = np.ones((self.height,50,3),dtype=np.uint8)*253
        img_cat = np.concatenate((img1,buffer),axis=1)
        img_cat = np.concatenate((img_cat,img2),axis=1)
        cv.imshow(left + " | " + right, img_cat)
        cv.moveWindow(left + " | " + right, 100,100)
        k = cv.waitKey(0)
        cv.destroyAllWindows()


    def generateDispMap(self, windowSize):
        dispMap = np.zeros(self.imgL.shape[0:2])
        # for y in range(200,290):#(100,300):#self.width):
        #     for x in range(450,540):#(100,300):#self.height):
        for y in range(0,self.height):#(100,300):#self.width):
            print(y)
            for x in range(0,self.width):#(100,300):#self.height):
                curPoint = np.array([y,x])
                # t = time.time()
                corresPoint = self.findCorrespondant(x,y, windowSize)
                # elapsed1 = time.time() - t
                # dist = abs(np.linalg.norm(curPoint - corresPoint))
                # t = time.time()
                # dist = np.sqrt((curPoint[0]-corresPoint[0])**2 + (curPoint[1]-corresPoint[1])**2)
                dist = curPoint[0]-corresPoint[0]
                # elapsed2 = time.time() -t
                dispMap[y,x] = abs(dist)
                # print([elapsed1])

        # dispMap = np.array(dispMap, dtype=np.uint8)
        # max = np.max(dispMap)
        mean = np.mean(dispMap)
        std = np.std(dispMap)

        if std != 0:
            dispMap = (dispMap - mean)/std
            minDisp = np.min(dispMap)
            maxDisp = np.max(dispMap)
            dispMap = (dispMap - minDisp) * (254 - 0) / (maxDisp - minDisp) + 0 #map
        else:
            dispMap = dispMap - mean

        # if max != 0:
        #     dispMap = (dispMap / max) * 255
        

        dispMap = np.array(dispMap,dtype=np.uint8)
        elapsed1 = time.time() - t
        print(elapsed1)
        cv.imshow("dispMap", dispMap)
        k = cv.waitKey(0)
        







#################################################################
if __name__ == "__main__":
    dirname = os.path.dirname(__file__)
    imagLpath = dirname + "\\MiddEval3-data-Q\\MiddEval3\\trainingQ\\Motorcycle\\im0.png"
    imagRpath = dirname + "\\MiddEval3-data-Q\\MiddEval3\\trainingQ\\Motorcycle\\im1.png"
    cam0=np.array([[999.421, 0, 294.182], [0, 999.421, 252.932], [0, 0, 1,]])
    cam1=np.array([[999.421, 0, 326.96], [0, 999.421, 252.932], [0, 0, 1]])
    width=741
    height=497
    doffs=32.778
    baseline=193.001

    s = stereoSystem(imagLpath, imagRpath, cam0, cam1, height, width, doffs, baseline)
    # s.drawEpipolar(100,200)
    # s.findFundMatr()
    # corresPoint = s.findCorrespondant(200,200, 101)
    # s.displayCorrespondent(150,150, 31)

    # t = time.time()
    # s.generateDispMap(23)
    


dirname = os.path.dirname(__file__)
imagLpath = dirname + "\\MiddEval3-data-Q\\MiddEval3\\trainingQ\\Pipes\\im0.png"
imagRpath = dirname + "\\MiddEval3-data-Q\\MiddEval3\\trainingQ\\Pipes\\im1.png"
cam0=np.array([[989.886, 0, 392.942], [0, 989.886, 243.221], [0, 0, 1,]])
cam1=np.array([[989.886, 0, 412.274], [0, 989.886, 243.221], [0, 0, 1]])
width=735
height=485
doffs=19.331
baseline=236.922

s2 = stereoSystem(imagLpath, imagRpath, cam0, cam1, height, width, doffs, baseline)
# s2.drawEpipolar(100,200)
# s2.findFundMatr()
# corresPoint = s.findCorrespondant(200,200, 101)
# s2.displayCorrespondent(150,150, 27)
# s2.displayCorrespondent(250,350, 27)
# s2.displayCorrespondent(350,250, 27)
# s2.displayCorrespondent(365,241, 27)
# s2.displayCorrespondent(600,50, 27)
t = time.time()
s.generateDispMap(23)