import cv2
import numpy as np
import os
import glob
from matplotlib import pyplot as plt

# Defining the dimensions of checkerboard
CHECKERBOARD = (6,9)

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# cv2.TERM_CRITERIA_EPS:指定された精度(epsilon)に到達したら繰り返し計算を終了する
# cv2.TERM_CRITERIA_MAX_ITER:指定された繰り返し回数(max_iter)に到達したら繰り返し計算を終了する
# cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER : 上記のどちらかの条件が満たされた時に繰り返し計算を終了する
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Defining the world coordinates for 3D points
objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
prev_img_shape = None

file_name = glob.glob('"C:\\Users\\t0725\\Downloads\\learnopencv-master\\learnopencv-master\\CameraCalibration\\images\\image_5.jpg"')[0]
img = cv2.imread(file_name)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# Find the chess board corners
# If desired number of corners are found in the image then ret = true
ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)

# Creating vector to store vectors of 3D points for each checkerboard image
objpoints = []

# Creating vector to store vectors of 2D points for each checkerboard image
imgpoints = [] 

if ret == True:
    objpoints.append(objp)
    
    # refining pixel coordinates for given 2d points.
    corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
    
    imgpoints.append(corners2)
    
    # Draw and display the corners
    img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2,ret)
    

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) # OpenCV は色がGBR順なのでRGB順に並べ替える
plt.show()
