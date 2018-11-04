from scipy.spatial import distance as dist
import numpy as np
import mahotas
import cv2
import imutils
import argparse


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())

# load the reference image containing the object we want to detect,
refImage = cv2.imread(args["image"])
gray = cv2.cvtColor(refImage, cv2.COLOR_BGR2GRAY)

moments = mahotas.features.zernike_moments(gray.copy(), 200, degree=3)
print(f"Features: {moments}")
