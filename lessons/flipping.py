import argparse
import imutils
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())

# load the image and show it
image = cv2.imread(args["image"])
cv2.imshow("Original", image)

# flip the image horizontally
flipped = cv2.flip(image, 1)
cv2.imshow("Flipped Horizontally", flipped)

# check a pixel
(b, g, r) = flipped[235, 259]
print(f"Pixel at (259, 235) - Red: {r}, Green: {g}, Blue: {b}")

# rotate the image 45 degrees counter-clockwise
rotated = imutils.rotate(flipped, 45)

# flip the image vertically
flipped = cv2.flip(rotated, 0)

# check a pixel
(b, g, r) = flipped[189, 441]
print(f"Pixel at (441, 189) - Red: {r}, Green: {g}, Blue: {b}")

cv2.imshow("Flipped Vertically", flipped)

# flip the image along both axes
flipped = cv2.flip(image, -1)
cv2.imshow("Flipped Horizontally and Vertically", flipped)
cv2.waitKey(0)