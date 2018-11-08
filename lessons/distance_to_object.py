from markers.distancefinder import DistanceFinder
from imutils import paths
import argparse
import imutils
import cv2


ap = argparse.ArgumentParser()
ap.add_argument("-r", "--reference", required=True, help="Path to the reference image")
ap.add_argument("-w", "--ref-width-inches", required=True, type=float, help="reference object in inches")
ap.add_argument("-d", "--ref-distance-inches", required=True, type=float, help="distance to reference object in inches")
ap.add_argument("-i", "--images", required=True, help="path tot he directory containing images to test")
args = vars(ap.parse_args())

# load the reference image and resize it
refImage = cv2.imread(args["reference"])
refImage = imutils.resize(refImage, height=700)

# initialize the distance finder, find the marker in the image, and calibrate the distance
# finder
df = DistanceFinder(args["ref_width_inches"], args["ref_distance_inches"])
refMarker = DistanceFinder.findSquareMarker(refImage)
df.calibrate(refMarker[2])

# visualize the results on the reference image and display it
refImage = df.draw(refImage, refMarker, df.distance(refMarker[2]))
cv2.imshow("Reference", refImage)

for imagePath in paths.list_images(args["images"]):
    filename = imagePath[imagePath.rfind('/') + 1:]
    image = cv2.imread(imagePath)
    image = imutils.resize(image, height=700)
    print(f"[INFO] processing {filename}")

    # find the marker in the image
    marker = DistanceFinder.findSquareMarker(image)

    # if the marker is None, then the square marker could not be found in the image
    if marker is None:
        print(f"[INFO] could not find marker for {filename}")
        continue

    # determine the distance to the marker
    distance = df.distance(marker[2])

    # visualize the results on the image and display it
    image = df.draw(image, marker, distance)
    cv2.imshow("image", image)
    cv2.waitKey(0)