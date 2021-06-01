import numpy as np
import cv2 as cv
import os
import copy
from matplotlib import pyplot as plt

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

        img1 = cv.cvtColor(self.imgL, cv.COLOR_BGR2GRAY)
        img2 = cv.cvtColor(self.imgR, cv.COLOR_BGR2GRAY)

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
    s.findFundMatr()