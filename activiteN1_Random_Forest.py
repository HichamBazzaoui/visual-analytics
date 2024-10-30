

import numpy as np
import os

# On va nommer le dossier par le nom de la classe
train_path = "../images/dataset/train"
training_names = os.listdir(train_path)
image_paths = []
image_classes = []
class_id = 0


def imglist(path):
    return [os.path.join(path, f) for f in os.listdir(path)]


for training_name in training_names:
    dir = os.path.join(train_path, training_name)
    print("Processing directory:", dir)
    class_path = imglist(dir)
    image_paths += class_path
    image_classes += [class_id] * len(class_path)
    class_id += 1

print("Step 1: Extracting descriptors using SIFT and stacking them into a descriptor vector. Resizing all images to 128*128.")
from skimage.feature import SIFT
from PIL import Image

des_list = []  # Create a list to store all descriptors
descriptor_extractor = SIFT()
for image_path in image_paths:
    im = np.array(Image.open(image_path).convert("L").resize((128, 128)))
    descriptor_extractor.detect_and_extract(im)
    kpts = descriptor_extractor.keypoints
    des = descriptor_extractor.descriptors
    des_list.append((image_path, des))
print("Descriptors extraction completed.")

# Stack all descriptors vertically into a numpy array
descriptors = des_list[0][1]
for image_path, descriptor in des_list[1:]:
    descriptors = np.vstack((descriptors, descriptor))

print("Step 2: Creating the Bag-Of-Features (BOF)")
# kmeans only works on float types
descriptors_float = descriptors.astype(float)

# Perform k-means clustering and vector quantization
from scipy.cluster.vq import kmeans, vq

k = 200  # k-means with 200 clusters
codebook, variance = kmeans(descriptors_float, k, 1)
print("K-means clustering completed.")

# Calculate the feature histogram and represent them as a vector
# vq Assigns codes from the codebook to observations
im_features = np.zeros((len(image_paths), k), "float32")
for i in range(len(image_paths)):
    words, distance = vq(des_list[i][1], codebook)
    for w in words:
        im_features[i][w] += 1

print("Feature histogram calculation completed.")
# Normalize the features by removing the mean and scaling to unit variance
from sklearn.preprocessing import StandardScaler

stdSlr = StandardScaler().fit(im_features)
im_features = stdSlr.transform(im_features)

print("Step 3: Training the Random forest model")
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators = 100, random_state=30)
clf.fit(im_features, np.array(image_classes))
print("random forest training completed.")
# Save the SVM model
# Joblib dumps the Python object into a single file
import joblib

joblib.dump((clf, training_names, stdSlr, k, codebook), "bof_Random_forest.pkl", compress=3)
print("Model saved as 'bof_Random_forest.pkl'")
