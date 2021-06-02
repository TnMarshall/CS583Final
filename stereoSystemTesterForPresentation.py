import os
import numpy as np
from stereoSystem import stereoSystem

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
# corresPoint = s.findCorrespondant(200,200, 101)
s.displayCorrespondent(150,150, 31)
s.displayCorrespondent(250,350, 31)
s.displayCorrespondent(350,250, 31)
s.displayCorrespondent(365,241, 31)
s.displayCorrespondent(600,50, 31)
# s.generateDispMap(31)

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
s2.findFundMatr()
# corresPoint = s.findCorrespondant(200,200, 101)
s2.displayCorrespondent(150,150, 31)
s2.displayCorrespondent(250,350, 31)
s2.displayCorrespondent(350,250, 31)
s2.displayCorrespondent(365,241, 31)
s2.displayCorrespondent(600,50, 31)
# s.generateDispMap(31)