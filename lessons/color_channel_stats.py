from scipy.spatial import distance as dist
from imutils import paths
import numpy as np
import argparse
import cv2
import os

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--path", required=True, help="Path to the images")
args = vars(ap.parse_args())

# grab the image path and initialize the index to store the image filename
# and feature vector
imagePaths = sorted(list(paths.list_images(args["path"])))
index = {}

for imagePath in imagePaths:
    image = cv2.imread(imagePath)
    filename = os.path.basename(imagePath)

    # extract the mean and standard deviation from each channel of the
    # BGR image, then update the index with the feature vector
    (means, stds) = cv2.meanStdDev(image)
    features = np.concatenate([means, stds]).flatten()
    index[filename] = features

# display hte query image and grab the sorted keys of the index dictionary
query = cv2.imread(imagePaths[0])
cv2.imshow(f"Query {os.path.basename(imagePaths[0])}", query)
keys = sorted(index.keys())[1:]

# loop over the filenames in the dictionary
for (i, k) in enumerate(keys):
    # load the current image and compute the Euclidean distance between the
    # query image (i.e. the first image) and the current image
    image = cv2.imread(imagePaths[i+1])
    d = dist.euclidean(index["trex_01.png"], index[k])

    # display the distance between the query image and the current image
    cv2.putText(image, "%.2f" % (d), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    cv2.imshow(k, image)

cv2.waitKey(0)