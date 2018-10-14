import cv2

image = cv2.imread("../images/oriented/orientation_example.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# load the face detector and detect faces in the image
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
rects = detector.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=9, 
    minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

# loop over the faces and draw a rectangle surrounding each
for (x, y, w, h) in rects:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# show detected faces
cv2.imshow("Faces", image)
cv2.waitKey(0)