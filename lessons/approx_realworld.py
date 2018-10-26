import argparse
import cv2
import imutils

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())

# load the image, convert it to grayscale
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
edged = cv2.Canny(gray, 75, 200)

cv2.imshow("Original", image)
cv2.imshow("Edge Map", edged)

# find contours in the image and sort them from largest to smallest,
# keeping only the largest ones
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:7]

# loop over the contours
for c in cnts:
    # approximate the contour and initialize the contour color
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.01 * peri, True)

    # show the difference in number of vertices between the original
    # and approximated contours
    print(f"original: {len(c)}, approx: {len(approx)}")

    # if the approximated contour has 4 vertices, then we have found
    # our rectangle
    if len(approx) == 4:
        cv2.drawContours(image, [approx], -1, (0, 255, 0), 2)
        
cv2.imshow("image", image)
cv2.waitKey(0)