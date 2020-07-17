# ------------------------------------------------------------------
#  Code to train a model to perform segmentation using Random Forest
#  and SVM classifiers
#  Autor: Rafael Vianna
#  Date: 2020 July 
# -----------------------------------------------------------------

import numpy as np
import cv2
from features import extract_feature
import time
import pandas as pd

# ------------------------------------------------------------------
# -------------- Import Images and Create DataFrame ----------------
# ------------------------------------------------------------------
name = "gfrp"

# Import original image
# ------------------------------------------------------------------
img = cv2.imread(name +'.tif')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Feature Extraction
# ------------------------------------------------------------------
df=extract_feature(img)

# Import labeled image and add to dataframe
# ------------------------------------------------------------------
labeled = cv2.imread('labeled_'+ name +'.tif')
labeled = cv2.cvtColor(labeled, cv2.COLOR_BGR2GRAY)
labeled_array = labeled.reshape(-1)         
df['Label']= labeled_array     

# ------------------------------------------------------------------
# ---------------------- Training and Validation -------------------
# ------------------------------------------------------------------

# Dependent variables
# ------------------------------------------------------------------
Y= df['Label'].values

# Independent variables
# ------------------------------------------------------------------
X= df.drop(labels = ['Label'], axis=1)

# Classifiers
# ------------------------------------------------------------------
# Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
rfc_model = RandomForestClassifier(n_estimators=10)
# Suport Vector Classifier
from sklearn.svm import LinearSVC
svc_model = LinearSVC(max_iter=100, tol=1e-2, multi_class='crammer_singer', loss='hinge', C=1)
# Logistic RegressionClassifier
from sklearn.linear_model import LogisticRegression
log_model=LogisticRegression(max_iter=100, tol=1e-2, solver='newton-cg', C=10)
# Naive Bayes Classifier
from sklearn.naive_bayes import GaussianNB
gnb_model = GaussianNB()


# Stratified 10-fold
# ------------------------------------------------------------------
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

accuracy_rfc=[]
accuracy_svc=[]
accuracy_log=[]
accuracy_gnb=[]
time_rfc=[]
time_svc=[]
time_log=[]
time_gnb=[]

skf=StratifiedKFold(n_splits=10)
skf.get_n_splits(X,Y)
for train_index, test_index in skf.split(X,Y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    Y_train, Y_test = np.ravel(pd.DataFrame(Y).iloc[train_index]), np.ravel(pd.DataFrame(Y).iloc[test_index])
    
    # RFC
    start=time.time()
    rfc_model.fit(X_train, Y_train)
    Y_pred = rfc_model.predict(X_test)
    score=accuracy_score(Y_pred, Y_test)
    accuracy_rfc.append(score)
    time_rfc.append(time.time()-start)
    
    # SVC
    start=time.time()
    svc_model.fit(X_train, Y_train)
    Y_pred = svc_model.predict(X_test)
    score=accuracy_score(Y_pred, Y_test)
    accuracy_svc.append(score)
    time_svc.append(time.time()-start)
    
    # Logistic Regression
    start=time.time()
    log_model.fit(X_train, Y_train)
    Y_pred = log_model.predict(X_test)
    score=accuracy_score(Y_pred, Y_test)
    accuracy_log.append(score)
    time_log.append(time.time()-start)
    
    # SVC
    start=time.time()
    gnb_model.fit(X_train, Y_train)
    Y_pred = gnb_model.predict(X_test)
    score=accuracy_score(Y_pred, Y_test)
    accuracy_gnb.append(score)
    time_gnb.append(time.time()-start)

print("RFC: ")
print("    Accuracy = ", np.array(accuracy_rfc).mean())
print("    Average Time = ", np.array(time_rfc).mean())
print("SVC: ")
print("    Accuracy = ", np.array(accuracy_svc).mean())
print("    Average Time = ",np.array(time_svc).mean())
print("Log_Reg: ")
print("    Accuracy = ", np.array(accuracy_log).mean())
print("    Average Time = ", np.array(time_log).mean())
print("N-B: ")
print("    Accuracy = ", np.array(accuracy_gnb).mean())
print("    Average Time = ", np.array(time_gnb).mean())


# Wilcoxon Test
# ------------------------------------------------------------------
# from scipy.stats import wilcoxon
# w,p = wilcoxon(np.array(accuracy_rfc),np.array(accuracy_svc))
# print(w,p)