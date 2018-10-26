import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())

# load the image, convert it to grayscale
image = cv2.imread(args["image"])
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# apply histogram equalization to stretch the contrast of our image
eq = cv2.equalizeHist(image)

cv2.imshow("Original", image)
cv2.imshow("Histogram Equalization", eq)
cv2.waitKey(0)

print(f"Value of pixel at (146, 272): {eq[272, 146]}")