# ------------------------------------------------------------------
#  Code to perform image segmentation using machine learning algorithms
#  Autor: Rafael Vianna
#  Date: 2020 July
# -----------------------------------------------------------------


import pickle
import cv2
import matplotlib.pyplot as plt
from features import extract_feature
import time

# Import original image
# ------------------------------------------------------------------
name = "gfrp"

# Import original image
# ------------------------------------------------------------------
img = cv2.imread(name + '.tif')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Feature Extraction
# ------------------------------------------------------------------
start = time.time()
X = extract_feature(img)
print("Feature Extraction time = ", time.time()-start)

# Load Model
# ------------------------------------------------------------------
filename = name + '_RF_classifier'
model = pickle.load(open(filename, 'rb'))  # Read binary file

# Classification
# ------------------------------------------------------------------
start = time.time()
classification = model.predict(X)
print("Segmentation time = ", time.time()-start)

# Show image
segmented_image = classification.reshape((img.shape))
plt.imshow(segmented_image)

# Save into a file
plt.imsave('segmented_'+name+'_RF.jpg', segmented_image)
