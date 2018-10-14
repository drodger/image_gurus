import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
ap.add_argument("-o", "--output", required=True, help="Filename to save the image as")
args = vars(ap.parse_args())

# load the iamge and display some basic information on it
image = cv2.imread(args["image"])
print(f"shape: {image.shape}")
print(f"width: {image.shape[1]}")
print(f"height: {image.shape[0]}")
print(f"channels: {image.shape[2]}")

cv2.imshow("Image", image)
cv2.waitKey(0)

# save the image -- OpenCV handles converting the filetypes
cv2.imwrite(args["output"], image)