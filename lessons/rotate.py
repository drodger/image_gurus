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

# grab the dimensions of the shape and calculate the center of the image
(h, w) = image.shape[:2]
(cX, cY) = (w / 2, h / 2)

# rotate our image by 45 degrees
M = cv2.getRotationMatrix2D((cX, cY), 45, 1.0)
rotated = cv2.warpAffine(image, M, (w, h))
cv2.imshow("Rotated by 45 Degrees", rotated)

# rotate image by -90 degrees
M = cv2.getRotationMatrix2D((cX, cY), -90, 1.0)
rotated = cv2.warpAffine(image, M, (w, h))
cv2.imshow("Rotated by -90 Degrees", rotated)

# rotate the image around an arbitrary point rather than the center
M = cv2.getRotationMatrix2D((cX - 50, cY - 50), 45, 1.0)
rotated = cv2.warpAffine(image, M, (w, h))
cv2.imshow("Rotated by Offset & 45 Degrees", rotated)
cv2.waitKey(0)

# rotate image 30 degrees clockwise (negative) using imutils.rotate()
rotated = imutils.rotate(image, -30)

# check a pixel
(b, g, r) = rotated[254, 335]
print(f"Pixel at (335, 254) - Red: {r}, Green: {g}, Blue: {b}")

# rotate image 110 degrees counter-clockwise
rotated = imutils.rotate(image, 110)

# check a pixel
(b, g, r) = rotated[136, 312]
print(f"Pixel at (312, 136) - Red: {r}, Green: {g}, Blue: {b}")

# rotate image 88 degrees counter-clockwise, about coordinate (50, 50)
rotated = imutils.rotate(image, 88, (50, 50))

# check a pixel
(b, g, r) = rotated[10, 10]
print(f"Pixel at (10, 10) - Red: {r}, Green: {g}, Blue: {b}")