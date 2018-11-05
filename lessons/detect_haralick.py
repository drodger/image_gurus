import argparse
import mahotas
import cv2


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())

# load the reference image containing the object we want to detect
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

features = mahotas.features.haralick(gray).mean(axis=0)
print(f"Features: {features}")