import numpy as np
import argparse
import imutils
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())

# load the image and show it
image = cv2.imread(args["image"])
cv2.imshow("Original", image)

# we need to keep in mind aspect ratio so the image does not look skewed
# or distorted -- therefore, we ca clulate the ratio of the new image to
# the old image. Let's make our new image have a width of 150 pixels
r = 150.0 / image.shape[1]
dim = (150, int(image.shape[0] * r))

# perform the actual resizing
resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
cv2.imshow("Resized (Width)", resized)

# We can apply the same concept if we wanted to adjust the height of the image
# but instead calculating the ratio based on height -- let's make the height of
# the rezied image 50 pixels
r = 50.0 / image.shape[0]
dim = (int(image.shape[1] * r), 50)

# perform the resizing
resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
cv2.imshow("Resized (Height)", resized)

# Now use imutils.resize
resized = imutils.resize(image, width=100)
cv2.imshow("Resized via imutils", resized)
cv2.waitKey(0)

# construct list of interpolation methods
methods = [
    ("cv2.INTER_NEAREST", cv2.INTER_NEAREST),
    ("cv2.INTER_LINEAR", cv2.INTER_LINEAR),
    ("cv2.INTER_AREA", cv2.INTER_AREA),
    ("cv2.INTER_CUBIC", cv2.INTER_CUBIC),
    ("cv2.INTER_LANCZOS4", cv2.INTER_LANCZOS4),
]

# loop over the interpolation methods
for (name, method) in methods:
    # increase size of image by 3x using current interpolation method
    resized = imutils.resize(image, width=image.shape[1] * 3, inter=method)
    cv2.imshow(f"Method: {name}", resized)
    cv2.waitKey(0)

# Resize small image to have a width of 100px, using INTER_NEAREST
resized = imutils.resize(image, width=100, inter=cv2.INTER_NEAREST)

# check a pixel
(b, g, r) = resized[74, 20]
print(f"Pixel at (20, 74) - Red: {r}, Green: {g}, Blue: {b}")

# Make image 2x, using INTER_CUBIC
resized = imutils.resize(image, width=image.shape[1] * 2, inter=cv2.INTER_CUBIC)

# check a pixel
(b, g, r) = resized[367, 170]
print(f"Pixel at (170, 367) - Red: {r}, Green: {g}, Blue: {b}")