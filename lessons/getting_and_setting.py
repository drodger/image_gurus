import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())

# load the image,  grab its dimensions, and show it
image = cv2.imread(args["image"])
(h, w) = image.shape[:2]
cv2.imshow("Original", image)

# images are just numpy arrays. The top-left pixel can be found at (0, 0)
(b, g, r) = image[0, 0]
print(f"Pixel at (0, 0) - Red: {r}, Green: {g}, Blue: {b}")

# let's change the value of the pixel at (0, 0) and make it red
image[0, 0] = (0, 0, 255)
(b, g, r) = image[0, 0]
print(f"Pixel at (0, 0) - Red: {r}, Green: {g}, Blue: {b}")

# compute the center of the image
(cX, cY) = (w // 2, h // 2)

# since we are using numpy arrays, we can aplpy slicing and grab large chunks
# of the image -- let's grab the top-left corner
tl = image[0:cY, 0:cX]
cv2.imshow("Top-Left corner", tl)

# in a similar fashio, let's grab the top-right, bottom-right, and bottom-left
# corners and display them
tr = image[0:cY, cX:w]
br = image[cY:h, cX:w]
bl = image[cY:h, 0:cX]
cv2.imshow("Top-Right corner", tr)
cv2.imshow("Bottom-Right corner", br)
cv2.imshow("Bottom-Left corner", bl)

# make the top-left corner of the original image green
image[0:cY, 0:cX] = (0, 255, 0)

# show the updated image
cv2.imshow("Updated", image)
cv2.waitKey(0)

(b, g, r) = image[225, 111]
print(f"Pixel at (111, 225) - Red: {r}, Green: {g}, Blue: {b}")