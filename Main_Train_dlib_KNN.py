import math
from sklearn import neighbors
import os
import os.path
import pickle
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder
import cv2

"""""
    Trains a k-nearest neighbors classifier for face recognition.
    :param train_dir: directory that contains a sub-directory for each known person, with its name.
     (View in source code to see train_dir example tree structure)
     Structure:
        <train_dir>/
        ├── <person1>/

        │   ├── <somename1>.jpeg
        │   ├── <somename2>.jpeg
        │   ├── ...
        ├── <person2>/
        │   ├── <somename1>.jpeg
        │   └── <somename2>.jpeg
        └── ...
"""""

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
# trainDir = "KNN_Train"
trainDir = "KNN_Train_50"
modelSaveName = trainDir + "_model.clf"

n_neighbors = 2
knn_algo = 'ball_tree'
weights = 'distance'

def mainTrain(n_neighbors, knn_algo, weights):
    
    print("Training KNN classifier...")

    X = []
    y = []

    for classDir in os.listdir(trainDir):
        print(classDir)
        for imgPath in image_files_in_folder(os.path.join(trainDir, classDir)):
            img = cv2.imread(imgPath)
            image = img.copy()
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            faceBboxes = face_recognition.face_locations(image)

            if len(faceBboxes) != 1:
                print("There is no face in training image")
            else:
                encodingsFace = face_recognition.face_encodings(image, faceBboxes)[0]
                # print(encodingsFace)
                X.append(encodingsFace)
                y.append(classDir)

    if n_neighbors is None:
        n_neighbors = int(round(math.sqrt(len(X))))
        print("Chose n_neighbors automatically:", n_neighbors)

    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights=weights)
    knn_clf.fit(X, y)

    if modelSaveName is not None:
        with open(modelSaveName, 'wb') as f:
            pickle.dump(knn_clf, f)
        print(modelSaveName, '=> Training Done')


if __name__ == "__main__":
   mainTrain(n_neighbors, knn_algo, weights)