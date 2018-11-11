import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-s", "--source", required=True, help="Path to the source image")
ap.add_argument("-t", "--template", required=True, help="Path to the template image")
args = vars(ap.parse_args())

source = cv2.imread(args["source"])
template = cv2.imread(args["template"])
(tempH, tempW) = template.shape[:2]

result = cv2.matchTemplate(source, template, cv2.TM_CCOEFF)
(minVal, MaxVal, minLoc, (x, y)) = cv2.minMaxLoc(result)
print(f"Found at ({x},{y})")
# draw a bounding box around the found image

cv2.rectangle(source, (x, y), (x + tempW, y + tempH), (0, 0, 0), 5)
cv2.putText(source, "FOUND", (x + tempW, y + tempH), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 0), 3)


cv2.imshow("Source", source)
cv2.imshow("Template", template)
cv2.waitKey(0)