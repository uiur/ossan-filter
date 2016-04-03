import cv2
import sys

path = sys.argv[1]

cascade_file = "/usr/local/Cellar/opencv/2.4.12_2/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml"

image = cv2.imread(path)
cascade = cv2.CascadeClassifier(cascade_file)
faces = cascade.detectMultiScale(image,
                                 scaleFactor = 1.1,
                                 minNeighbors = 5,
                                 minSize = (28, 28))


for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

flag, buf = cv2.imencode('.png', image)
print(buf.tobytes())
