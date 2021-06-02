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
s.drawEpipolar(100,200)
s.findFundMatr()
# corresPoint = s.findCorrespondant(200,200, 101)
s.displayCorrespondent(150,150, 31)
# s.generateDispMap(31)