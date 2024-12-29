import numpy as np
import cv2 as cv
import pickle
import glob
import os

# FIND CHESSBOARD CORNERS - OBJECT POINTS AND IMAGE POINTS
chessboardSize = (7, 4)  # Size of the chessboard
frameSize = (640, 480)  # Size of the frames

# Termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points, like (0,0,0), (1,0,0), (2,0,0), ..., (8,5,0)
objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboardSize[0], 0:chessboardSize[1]].T.reshape(-1, 2)

# Size of chessboard squares in mm
size_of_chessboard_squares_mm = 20
objp = objp * size_of_chessboard_squares_mm

# Arrays to store object points and image points from all the images
objpoints = []  # 3D points in real-world space
imgpoints = []  # 2D points in image plane

# Load images
images = glob.glob('./images/*.png')
print(f"Found {len(images)} images.")

for image in images:
    img = cv.imread(image)
    if img is None:
        print(f"Failed to load image: {image}")
        continue
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv.findChessboardCorners(gray, chessboardSize, None)

    # If found, add object points and image points (after refining them)
    if ret:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)  # Use refined corners

        # Draw and display the corners
        cv.drawChessboardCorners(img, chessboardSize, corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(1000)
    else:
        print(f"Chessboard not found in image: {image}")

cv.destroyAllWindows()

# CALIBRATION
custom_directory = './'

# Calibrate the camera
ret, cameraMatrix, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, frameSize, None, None)

if ret:
    # Save the camera calibration result for later use
    with open(os.path.join(custom_directory, "cameraMatrix.pkl"), "wb") as f:
        pickle.dump(cameraMatrix, f)
    with open(os.path.join(custom_directory, "dist.pkl"), "wb") as f:
        pickle.dump(dist, f)
    
    print("Camera calibration was successful.")
else:
    print("Camera calibration failed.")
