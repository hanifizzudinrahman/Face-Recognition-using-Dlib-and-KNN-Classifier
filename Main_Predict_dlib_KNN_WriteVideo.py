import cv2
import face_recognition
import numpy as np
import os
import glob
import pickle

from Utils_FaceRecognition_dlib import showRectangleName, showFPS
import time

knn_clf = "KNN_Train_50_model.clf"
n_neighbors = 1
distance_threshold = 0.5

def main(knn_clf):

    with open(knn_clf, 'rb') as f:
        knn_clf = pickle.load(f)

    id = "Video 50 Images.mkv"
    # id = 0
    cap = cv2.VideoCapture(id)

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    out = cv2.VideoWriter("1Result_" + id, cv2.VideoWriter_fourcc('M','J','P','G'), 100, (frame_width,frame_height))

    pTime = time.time()
    while True:
        _, imgR = cap.read()
        img = imgR.copy()
        img = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        faceLoc = face_recognition.face_locations(img)

        if len(faceLoc) != 1:
            pass
        else:
            encodeFace = face_recognition.face_encodings(img, faceLoc)

            # Use the KNN model to find the best matches for the test face
            closest_distances = knn_clf.kneighbors(encodeFace, n_neighbors)
            are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(faceLoc))][0]

            # Predict classes and remove classifications that aren't within the threshold
            predict = knn_clf.predict(encodeFace)[0]
            # print(are_matches, closest_distances[0][0], predict, faceLoc)

            faceLoc = times(faceLoc[0], 4)
            if are_matches == True:
                #print(predict, faceLoc)
                imgR = showRectangleName(imgR, faceLoc, predict, color = (0,255,0))
            else:
                imgR = showRectangleName(imgR, faceLoc, "Unknown", color = (0,0,255))

        imgR, pTime = showFPS(imgR, pTime) 

        # Write the frame into the file
        out.write(imgR)
        cv2.imshow("imgPredict", imgR)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    cv2.destroyAllWindows()
    cap.release()
    out.release()


def times(faceVid, times):
    faceVid4 = list(faceVid)
    faceVid4_array = np.array(faceVid4)
    faceVid4_array = faceVid4_array * times
    faceVid = tuple(faceVid4_array)
    return faceVid

if __name__ == "__main__":
    main(knn_clf)