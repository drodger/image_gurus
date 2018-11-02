import argparse
import cv2
import imutils


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# compute the Hu Moments feature vector for the entire image and show it
# This is the *incorrect* way to compute Hu moments for an image with multiple
# objects. Instead of taking into account the multiple objects, Hu Moments
# are computed for the entire image. In this case, the centroid becomes the
# center of all three shapes, rather than just one of them.
moments = cv2.HuMoments(cv2.moments(image)).flatten()
print(f"ORIGINAL MOMENTS: {moments}")
cv2.imshow("Image", image)
cv2.waitKey(0)

# To correctly compute Hu Moments for each of the objects, we need to find the
# contours of each object, extract the ROI around each, and then compute the 
# Hu Moments for each ROI individually

# find the contours of each object
cnts = cv2.findContours(image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]

for (i, c) in enumerate(cnts):
    # extract the ROI from the image and compute the Hu Moments feature
    # vector for the ROI
    (x, y, w, h) = cv2.boundingRect(c)
    roi = image[y:y + h, x:x + w]
    moments = cv2.HuMoments(cv2.moments(roi)).flatten()

    print(f"MOMENTS FOR OBJECT #{i + 1}: {moments}")
    cv2.imshow("ROI ${i + 1}", roi)
    cv2.waitKey(0)