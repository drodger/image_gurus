from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from imutils import paths
import numpy as np
import argparse
import mahotas
import cv2
import sklearn


def describe(image):
    # extract the mean and stanbdard deviation from each channel of the image
    # in the HSV color space
    (means, stds) = cv2.meanStdDev(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    color_stats = np.concatenate([means, stds]).flatten()

    # extract Haralick texture features
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    haralick = mahotas.features.haralick(gray).mean(axis=0)

    # return a concatenated feature vector of color statistics and Haralick
    # texture features
    return np.hstack([color_stats, haralick])

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to 8 scene category dataset")
ap.add_argument("-f", "--forest", type=int, default=-1,
    help="whether or not Random Forest should be used")
args = vars(ap.parse_args())

# grab the set of image paths and initialize the list of labels and matrix of
# features
print("[INFO] extracting features...")
image_paths = sorted(paths.list_images(args["dataset"]))
labels = []
data = []

for image_path in image_paths:
    label = image_path[image_path.rfind("/") + 1:].split("_")[0]
    image = cv2.imread(image_path)

    features = describe(image)
    labels.append(label)
    data.append(features)

# construct the training and testing split by taking 75% of the data for training
# and 25% for testing
(train_data, test_data, train_labels, test_labels) = train_test_split(np.array(data),
    np.array(labels), test_size=0.25, random_state=42)

# initialize the model as a decision tree
model = DecisionTreeClassifier(random_state=84)

# check to see if a Random Forest should be used instead
if args["forest"] > 0:
    model = RandomForestClassifier(n_estimators=20, random_state=42)

# train the decision tree
print("[INFO] training model...")
model.fit(train_data, train_labels)

# evaluate the classifier
print("[INFO] evaluating...")
predictions = model.predict(test_data)
print(classification_report(test_labels, predictions))

for i in np.random.randint(0, high=len(image_paths), size=(10,)):
    image_path = image_paths[i]
    filename = image_path[image_path.rfind("/") + 1:]
    image = cv2.imread(image_path)
    features = describe(image)
    prediction = model.predict(features.reshape(1, -1))[0]

    print(f"[PREDICTION] {filename}: {prediction}")
    cv2.putText(image, prediction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv2.imshow("image", image)
    cv2.waitKey(0)