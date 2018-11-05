from local_binary_patterns.localbinarypatterns import LocalBinaryPatterns
from imutils import paths
import numpy as np
import argparse
import cv2


ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="Path to the dataset of images")
ap.add_argument("-q", "--query", required=True, help="Path to the query image")
args = vars(ap.parse_args())

# initialize the local binary patterns descriptor and initialize the index dictionary
# where the image filename is the key and the features are the value
desc = LocalBinaryPatterns(24, 8)
index = {}

# loop over the shirt images
for imagePath in paths.list_images(args["dataset"]):
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = desc.describe(gray)

    # update the index
    filename = imagePath[imagePath.rfind("/") + 1:]
    index[filename] = hist

# load the query iimage and extract Local Binary Patterns from it
query = cv2.imread(args["query"])
queryFeatures = desc.describe(cv2.cvtColor(query, cv2.COLOR_BGR2GRAY))

# show the query image and initialize the results dictionary
cv2.imshow("Query", query)
results = {}

# loop over the index
for (k, features) in index.items():
    # compute the chi-squared distance between the current features and the query
    # features, then update the dictionary of results
    d = 0.5 * np.sum(((features - queryFeatures) ** 2) / (features + queryFeatures + 1e-10))
    results[k] = d

# sort the results
results = sorted([(v, k) for (k, v) in results.items()])[:3]

for (i, (score, filename)) in enumerate(results):
    print(f"#{i + 1}. {filename}: {score:.4f}")
    image = cv2.imread(args["dataset"] + "/" + filename)
    cv2.imshow(f"Result #{i + 1}", image)
    cv2.waitKey(0)