# -*- coding: utf-8 -*-
import cv2
import sys
import os.path
import hashlib

def detect(filename, cascade_file = "../lbpcascade_animeface.xml", dirname = 'tmp/'):
    if not os.path.isfile(cascade_file):
        raise RuntimeError("%s: not found" % cascade_file)

    cascade = cv2.CascadeClassifier(cascade_file)
    image = cv2.imread(filename)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    faces = cascade.detectMultiScale(gray,
                                     # detector options
                                     scaleFactor = 1.1,
                                     minNeighbors = 5,
                                     minSize = (24, 24))
    i = 0
    for (x, y, w, h) in faces:
        m = hashlib.md5()
        m.update(open(filename).read())
        name = m.hexdigest() + '-' + str(i) + '.png'
    	cv2.imwrite(dirname + name, image[y:y+h, x:x+w])
        i += 1

if len(sys.argv) != 3:
    sys.stderr.write("usage: face.py <filename> <dir>\n")
    sys.exit(-1)

detect(sys.argv[1], cascade_file = "/usr/local/Cellar/opencv/2.4.12_2/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml", dirname=sys.argv[2])
