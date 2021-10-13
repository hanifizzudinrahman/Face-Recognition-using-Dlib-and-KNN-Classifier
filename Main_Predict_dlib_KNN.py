import math
from operator import pos
import os
import os.path
import pickle
import face_recognition
import cv2
from Utils_FaceRecognition_dlib import showRectangleName

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
# testDir = "KNN_Test"
testDir = "KNN_Test_50"
model_path = ""

# knn_clf = "KNN_Train_model.clf"
knn_clf = "KNN_Train_50_model.clf"
n_neighbors = 1
distance_threshold = 0.5

def mainPredict(imagePath, knn_clf, n_neighbors, distance_threshold):
    if not os.path.isfile(imagePath) or os.path.splitext(imagePath)[1][1:] not in ALLOWED_EXTENSIONS:
        raise Exception("Invalid image path: {}".format(imagePath))

    with open(knn_clf, 'rb') as f:
        knn_clf = pickle.load(f)

    imgR = cv2.imread(imagePath)
    img = imgR.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    imgBox = face_recognition.face_locations(img)

    # If no faces are found in the image, return an empty result.
    if len(imgBox) == 0:
        return []

    encodingsImage = face_recognition.face_encodings(img, imgBox)

    # Use the KNN model to find the best matches for the test face
    closest_distances = knn_clf.kneighbors(encodingsImage, n_neighbors)
    are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(imgBox))]

    # Predict classes and remove classifications that aren't within the threshold
    predict = knn_clf.predict(encodingsImage)
    print(are_matches, closest_distances[0][0], predict, imgBox)

    return imgR, are_matches[0], predict[0], imgBox[0]

if __name__ == "__main__":
   print("Predicting KNN classifier...") 

   for imageFile in os.listdir(testDir):
        imagePath = os.path.join(testDir, imageFile)
        print("Faces in: ", imageFile)
       
        img, match, name, position = mainPredict(imagePath, knn_clf, n_neighbors, distance_threshold)
        if match == True:
            # print(name, position)
            img = showRectangleName(img, position, name, color=(0,255,0))
        else:
            img = showRectangleName(img, position, "Unknown", color=(0,0,255))

        cv2.imshow("imgPredict", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()