import numpy as np
import cv2 as cv
import os

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
        self.picH = height
        self.picW = width
        self.hzOffset = hzoffset
        self.baseline = baseline
        self.C0 = np.array([self.Lintr[0,2],self.Lintr[1,2]])
        self.C1 = np.array([self.Rintr[0,2],self.Rintr[1,2]])
        self.f = self.Lintr[0,0]

        self.extrinsT = np.array([[baseline],[0],[0]])

        # cv.imshow("Left", self.imgL)
        # cv.imshow("Right", self.imgR)
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