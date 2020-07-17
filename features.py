#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 10:43:51 2020

@author: vianna
"""
import numpy as np
import pandas as pd
import cv2
from skimage.filters import gaussian, median, roberts, sobel, scharr, prewitt 
from skimage.filters.rank import entropy
from skimage.morphology import disk


def extract_feature(img):
    
    # Create Empty DataFrame
    # --------------------------------------------------------------
    df = pd.DataFrame()
    
    # Add original pixel value to the data as feature 1
    # --------------------------------------------------------------
    img_array = img.reshape(-1)         # Reshape picture into single collum
    df['Original Image']= img_array     # Include in dataframe
    
    # Add Entropy 
    # --------------------------------------------------------------
    entropy_img=entropy(img,disk(5))
    entropy_imgarray=entropy_img.reshape(-1)
    df['Entropy']= entropy_imgarray  
    
    # Add Gaussian (blur)
    # --------------------------------------------------------------
    img_gaussian = gaussian(img, sigma=3)
    img_gaussian_array = img_gaussian.reshape(-1)
    df['Gaussian']= img_gaussian_array 
    
    # Add Gaussian (blur)
    # --------------------------------------------------------------
    img_gaussian = gaussian(img, sigma=7)
    img_gaussian_array = img_gaussian.reshape(-1)
    df['Gaussian']= img_gaussian_array 
    
    # Add Gaussian (blur)
    # --------------------------------------------------------------
    img_gaussian = gaussian(img, sigma=10)
    img_gaussian_array = img_gaussian.reshape(-1)
    df['Gaussian']= img_gaussian_array 
    
    # Add Median (blur)
    # --------------------------------------------------------------
    img_median = median(img, np.ones((3,3)))
    img_median_array = img_median.reshape(-1)
    df['Median']= img_median_array 
    
    # Add Roberts Edge
    # --------------------------------------------------------------
    img_roberts = roberts(img)
    img_roberts_array = img_roberts.reshape(-1)
    df['Roberts']= img_roberts_array 
    
    # Add Sobel Edge
    # --------------------------------------------------------------
    img_sobel = sobel(img)
    img_sobel_array = img_sobel.reshape(-1)
    df['Sobel']= img_sobel_array 
    
    # Add Scharr Edge
    # --------------------------------------------------------------
    img_scharr = scharr(img)
    img_scharr_array = img_scharr.reshape(-1)
    df['Scharr']= img_scharr_array 
    
    # Add Prewitt Edge
    # --------------------------------------------------------------
    img_prewitt = prewitt(img)
    img_prewitt_array = img_prewitt.reshape(-1)
    df['Prewitt']= img_prewitt_array
    
    # Add Canny Edge
    # --------------------------------------------------------------
    img_canny = cv2.Canny(img, 100, 200)
    img_canny_array = img_canny.reshape(-1)
    df['Canny Edge']= img_canny_array
    
    #import matplotlib.pyplot as plt
    #plt.imshow(entropy_img)
    
    return df