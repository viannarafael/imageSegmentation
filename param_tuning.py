# ------------------------------------------------------------------
#  Code to determine best hyperparameters for image segmentation
#  Autor: Rafael Vianna
#  Date: 2020 July 
# -----------------------------------------------------------------

import pandas as pd
import cv2
from features import extract_feature
import time

# ------------------------------------------------------------------
# ------------------------ DataFrame Creation ----------------------
# ------------------------------------------------------------------
name = "gfrp"

# Import original image
# ------------------------------------------------------------------
img = cv2.imread(name +'.tif')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Feature Extraction
# ------------------------------------------------------------------
df=extract_feature(img)

#Import labeled image
# ------------------------------------------------------------------
labeled = cv2.imread('labeled_'+ name +'.tif')
labeled = cv2.cvtColor(labeled, cv2.COLOR_BGR2GRAY)
labeled_array = labeled.reshape(-1)         
df['Label']= labeled_array     

# Dependent variables
# ------------------------------------------------------------------
Y= df['Label'].values

# Independent variables
# ------------------------------------------------------------------
X= df.drop(labels = ['Label'], axis=1)

# ------------------------------------------------------------------
# ----------------------- Hyperparameter Tuning --------------------
# ------------------------------------------------------------------

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression


# Define the hyperparameter values for each model
# ------------------------------------------------------------------
model_params = {
#    'random_forest':{ 
#        'model': RandomForestClassifier(), 
#        'params': {'n_estimators': [5, 10, 15, 20, 25]}
#    }
#    'naive_bayes':{ 
#        'model': GaussianNB(), 
#        'params': {}
#    },
    'logistic_regression':{ 
        'model': LogisticRegression(max_iter=100, tol=1e-2), 
        'params': {'C': [1, 5, 10, 15, 20],
                   'solver': ['sag', 'saga', 'lbfgs', 'newton-cg']                   
                   }
    },
    'svc':{ 
        'model': LinearSVC(max_iter=100, tol=1e-2), 
        'params': {'C': [1, 5, 10, 15, 20],
                   'loss': ['hinge', 'squared_hinge'],
                   'multi_class': ['ovr', 'crammer_singer']
                   }
    }
   }

# Create a grid search for each model
# ------------------------------------------------------------------
scores=[]
 
for model_name, mp in model_params.items():
    print(model_name)
    start = time.time()
    clf = RandomizedSearchCV( mp['model'], mp['params'], cv=3, n_iter=10, n_jobs=-1)
    clf.fit(X,Y)
    scores.append({'model': model_name,
                   'best_score': clf.best_score_,
                   'best_params': clf.best_params_})
    print(model_name,"   ","Time for parameter tuning = ",time.time()-start)

# Print results  
result=pd.DataFrame(scores, columns=['model', 'best_score', 'best_params'])
print(result.head())