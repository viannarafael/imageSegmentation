# ------------------------------------------------------------------
#  Code to perform image segmentation using machine learning algorithms
#  Autor: Rafael Vianna
#  Date: 2020 July 
# -----------------------------------------------------------------
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from features import extract_feature
import time

# ------------------------------------------------------------------
# -------------------------- Import Images -------------------------
# ------------------------------------------------------------------

name = "gfrp"

# Import original image
# ------------------------------------------------------------------
img = cv2.imread(name +'.tif')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Import labeled image and add to dataframe
# ------------------------------------------------------------------
labeled = cv2.imread('labeled_'+ name +'.tif')
labeled = cv2.cvtColor(labeled, cv2.COLOR_BGR2GRAY)


# ------------------------------------------------------------------
# ------------------------ DataFrame Creation ----------------------
# ------------------------------------------------------------------

# Feature Extraction
# ------------------------------------------------------------------
df=extract_feature(img)


# Add labeled data
# ------------------------------------------------------------------
labeled_array = labeled.reshape(-1)         
df['Label']= labeled_array     


# ------------------------------------------------------------------
# ------------------------ Training Classifier ---------------------
# ------------------------------------------------------------------

# Dependent variables
# ------------------------------------------------------------------
Y= df['Label'].values

# Independent variables
# ------------------------------------------------------------------
X= df.drop(labels = ['Label'], axis=1)

# Split data into test and train
# ------------------------------------------------------------------
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state = 10)  


# Naive Bayes Classifier
# ------------------------------------------------------------------
start=time.time()
from sklearn.naive_bayes import GaussianNB
gnb_model = GaussianNB()
gnb_model.fit(X_train, Y_train)
print("Time of training= ",time.time()-start)

# ------------------------------------------------------------------
# -------------------------- Test Classifier -----------------------
# ------------------------------------------------------------------

# Prediction
# ------------------------------------------------------------------
Y_pred = gnb_model.predict(X_test)

# Metrics 
# ------------------------------------------------------------------
from sklearn import metrics
print("\nAcuracy =", metrics.accuracy_score(Y_test, Y_pred))
print(metrics.classification_report(Y_test, Y_pred))

# Classify image
# ------------------------------------------------------------------
start=time.time()
classification = gnb_model.predict(X)
print("Segmentation time = ",time.time()-start)
# Show image
segmented_image = classification.reshape((img.shape))
plt.imshow(segmented_image)



# ------------------------------------------------------------------
# -------------------------- Save Classifier -----------------------
# ------------------------------------------------------------------
import pickle

filename = name+'_NB_classifier'
pickle.dump(gnb_model, open(filename, 'wb')) # Save the model into a binary file
