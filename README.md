# Face-Recognition-using-Dlib-and-KNN-Classifier
![You Tube](https://user-images.githubusercontent.com/47806867/137105677-0f3d4134-f58c-4413-b1ff-d63d815626ca.png)

# YouTube: https://youtu.be/VvwEJaj5jKE

*Extract: 128 Features from Faces

This alghorithm useful when you wish to recognize a large set of known people,
and make a prediction for an unknown person in a feasible computation time.

Algorithm Description:
The knn classifier is first trained on a set of labeled (known) faces and can then predict the person
in an unknown image by finding the k most similar faces (images with closet face-features under euclidean distance)

in its training set, and performing a majority vote (possibly weighted) on their label.
For example, if k=3, and the three closest face images to the given image in the training set are one image of Biden and two images of Obama, The result would be 'Obama'.

* This implementation uses a weighted vote, such that the votes of closer-neighbors are weighted more heavily.

Detail using Dlib libraries for Face Recognition in here: https://medium.com/@ageitgey/machine-learning-is-fun-part-4-modern-face-recognition-with-deep-learning-c3cffc121d78
