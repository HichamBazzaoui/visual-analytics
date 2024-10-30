# -*- coding: utf-8 -*-
"""
Created on Thu May 30 09:32:00 2024

@author: Hicham
"""


import joblib
import numpy as np
import os
from PIL import Image
from skimage.feature import SIFT

#chargement de modele
clf, classes_names, stdSlr, k, voc = joblib.load("bof_KNN.pkl")

#On va nommer le dossier par le nom de la classe
test_path = '../images/dataset/test'
test_names = os.listdir(test_path)
image_paths = []
image_classes = []
class_id = 0
def imglist(path):    
    return [os.path.join(path, f) for f in os.listdir(path)]
for test_name in test_names:
    dir = os.path.join(test_path, test_name)
    class_path = imglist(dir)
    image_paths+=class_path
    image_classes+=[class_id]*len(class_path)
    class_id+=1

des_list = []	# Créer une liste où tous les descripteurs seront stockés
descriptor_extractor = SIFT()
charge = 0
image_paths_length = len(image_paths)
for index,image_path in enumerate(image_paths):
    im = np.array(Image.open(image_path).convert('L').resize((128,128)))
    descriptor_extractor.detect_and_extract(im)
    kpts = descriptor_extractor.keypoints
    des = descriptor_extractor.descriptors
    des_list.append((image_path, des))
    current_charge = int(100*(index+1)/image_paths_length)
    if current_charge != charge :
        charge = current_charge
        print(str(charge)+'%')
    
# Empiler tous les descripteurs verticalement dans un tableau numpy
descriptors = des_list[0][1]
for image_path, descriptor in des_list[1:]:
    descriptors = np.vstack((descriptors, descriptor))

#kmeans ne fonctionne que sur les types float
descriptors_float = descriptors.astype(float)  


# Effectuer le clustering k-means et la quantification vectorielle
from scipy.cluster.vq import kmeans, vq

codebook, variance = kmeans(descriptors_float, k, 1) 
# Calculer l'histogramme des caractéristiques et les représenter sous forme de vecteur
#vq Attribue des codes du codebook à des observations.
im_features = np.zeros((len(image_paths), k), "float32")
for i in range(len(image_paths)):
    words, distance = vq(des_list[i][1], codebook)
    for w in words:
        im_features[i][w] += 1
#Normaliser les features en supprimant la moyenne et en mettant à l'échelle la variance unitaire
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score,recall_score,f1_score,classification_report
stdSlr = StandardScaler().fit(im_features)
im_features = stdSlr.transform(im_features)


pred = clf.predict(im_features)
prediction =  [classes_names[i] for i in pred]
print(prediction)

accuracy = accuracy_score(image_classes, pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

cm = confusion_matrix(image_classes, pred)
print(f"confusion matrix:\n {cm}")

precision = precision_score(image_classes, pred,average="macro")
print(f"precision: {precision * 100:.2f}%")

recall = recall_score(image_classes, pred,average="macro")
print(f"recall: {recall * 100:.2f}%")

f1 = f1_score(image_classes, pred,average="macro")
print(f"f1: {f1 * 100:.2f}%")

print(classification_report(image_classes, pred))