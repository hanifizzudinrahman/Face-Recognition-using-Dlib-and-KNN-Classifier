from time import time
import cv2
import face_recognition
import time

def showRectangleName(imgR, faceVid, name, color = (0,0,255)):
    x1,x2,y2,y1 = faceVid
    cv2.rectangle(imgR, (x1,y1), (x2,y2), color, 2)
    cv2.putText(imgR, name, (x1+6, y2-6), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 1)
    return imgR

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

def showFPS(img, pTime):
    cTime = time.time()
    fps = 1/ (cTime - pTime)
    fps = round(fps,2)
    fps = "FPS: " + str(fps)
    cv2.putText(img, fps, (5,25), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
    return img, cTime