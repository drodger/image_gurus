import numpy as np
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())

# load the image and show it
image = cv2.imread(args["image"])
cv2.imshow("Original", image)

# images are NumPy arrays, stored as unsigned 8 bit integers -- this
# implies that the values of our pixels will be in the range [0, 255); when
# using functions like cv2.add and cv2.subtract, values will be clipped
# to this range, even if the added or subtracted values fall outside the
# range of [0, 255]. Example:
print(f"max of 255: {cv2.add(np.uint8([200]), np.uint8([100]))}")
print(f"min of 0: {cv2.subtract(np.uint8([50]), np.uint8([100]))}")

# NOTE: if you use NumPy arithmetic operations on these arrays, the value
# wil be modulo (wrap around) instead of being clipped to the [0, 255]
# range. This is important to keep in mind when working with images.
print(f"wrap around: {np.uint8([200]) + np.uint8([100])}")
print(f"wrap around: {np.uint8([50]) - np.uint8([100])}")

# let's increase the intensity of all pixels in our image by 100 -- we
# accomplish this by constructing a NumPy array that is teh same size of
# out matrix (filled with ones) and then multiplying it by 100 to create an
# array filled with 100's, then we simply add the images together; notice
# how the image is "brighter"
M = np.ones(image.shape, dtype="uint8") * 100
added = cv2.add(image, M)
cv2.imshow("Added", added)

# similarly, we can subtract 50 from all pixels in out image and make it
# darker
M = np.ones(image.shape, dtype="uint8") * 50
subtracted = cv2.subtract(image, M)
cv2.imshow("Subtracted", subtracted)
cv2.waitKey(0)

# add a value of 75 to all pixels, what is the value of the pixel
# at (61, 152)
M = np.ones(image.shape, dtype="uint8") * 75
added = cv2.add(image, M)
(b, g, r) = added[152, 61]
print(f"Pixel at (61, 152) - Red: {r}, Green: {g}, Blue: {b}")