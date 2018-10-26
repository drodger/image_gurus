import argparse
import cv2
import imutils

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())

# load the image, convert it to grayscale
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# find contours in the image
cnts = cv2.findContours(gray.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]

# loop over the contours
for c in cnts:
    # approximate the contour
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.01 * peri, True)

    # if the approximated contour has 4 vertices, then we are examining
    # a rectangle
    if len(approx) == 4:
        # draw the outline of the contour and draw the text on the image
        cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
        (x, y, w, h) = cv2.boundingRect(approx)
        cv2.putText(image, "Rectangle", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (0, 255, 255), 2)
        
cv2.imshow("image", image)
cv2.waitKey(0)