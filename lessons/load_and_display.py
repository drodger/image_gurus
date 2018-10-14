import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())

# load the iamge and display some basic information on it
image = cv2.imread(args["image"])
print(f"width: {image.shape[1]}")
print(f"height: {image.shape[0]}")
print(f"channels: {image.shape[2]}")

cv2.imshow("Image", image)
cv2.waitKey(0)

# save the image -- OpenCV handles converting the filetypes
cv2.imwrite("newimage.jpg", image)