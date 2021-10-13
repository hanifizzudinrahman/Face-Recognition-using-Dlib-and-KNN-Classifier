import cv2
import face_recognition
import numpy as np
import os
import glob
from pathlib import Path
from Utils_FaceRecognition_dlib import findEncodings, showRectangleName, showFPS
import time

realPath = "Images/Real/"
images = []
classNames = []


def main():
    for filename in glob.glob(realPath+"*.jpg"):
        curImg = cv2.imread(filename)
        images.append(curImg)
        classNames.append(Path(filename).stem)
    print(classNames)

    encodeListKnow = findEncodings(images)

    cap = cv2.VideoCapture(0)

    pTime = time.time()
    while True:
        _, imgR = cap.read()
        img = imgR.copy()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faceLoc = face_recognition.face_locations(img)
        encodeFace = face_recognition.face_encodings(img, faceLoc)

        for encodeVid, faceVid in zip(encodeFace, faceLoc):
            matches = face_recognition.compare_faces(encodeListKnow, encodeVid)
            faceDis = face_recognition.face_distance(encodeListKnow, encodeVid)

            matchIndex = np.argmin(faceDis)

            if matches[matchIndex]:
                name = classNames[matchIndex]
                imgR = showRectangleName(imgR, faceVid, name)

        imgR, pTime = showFPS(imgR, pTime)        
        cv2.imshow('vid', imgR)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    cv2.destroyAllWindows()
    cap.release()

if __name__ == "__main__":
    main()