import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())

# load the image and show it
image = cv2.imread(args["image"])
cv2.imshow("Original", image)

# cropping an image is accomplished using simple NumPy slices
face = image[85:250, 85:220]  # start y:end y, start x:end x
cv2.imshow("Face", face)
cv2.waitKey(0)

# crop entire body 
body = image[90:450, 0:290]
cv2.imshow("Body", body)
cv2.waitKey(0)

# crop people
# people = image[150:250, 0:85]
people = image[173:235, 13:81]
cv2.imshow("People", people)
cv2.waitKey(0)

# crop boat
boat = image[120:220, 225:380]
cv2.imshow("Boat", boat)
cv2.waitKey(0)