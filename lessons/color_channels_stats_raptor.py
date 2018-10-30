from scipy.spatial import distance as dist
from imutils import paths
import numpy as np
import argparse
import cv2
import os

ap = argparse.ArgumentParser()
ap.add_argument("-i1", "--image1", required=True, help="Path to image1")
ap.add_argument("-i2", "--image2", required=True, help="Path to image2")
args = vars(ap.parse_args())

# Extract the mean and standard deviation from each channel of the
# BGR image, then create and display the feature vector
image1 = cv2.imread(args["image1"])
image2 = cv2.imread(args["image2"])
index = {}

(means1, stds1) = cv2.meanStdDev(image1)
(means2, stds2) = cv2.meanStdDev(image2)
features1 = np.concatenate([means1, stds1]).flatten()
features2 = np.concatenate([means2, stds2]).flatten()
index["image1"] = features1
index["image2"] = features2

# Calculate the Euclidean distance between the two images
d = dist.euclidean(index["image1"], index["image2"])
print(f"Euclidean distance: {d}")