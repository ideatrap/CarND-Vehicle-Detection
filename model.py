import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
from skimage.feature import hog

'''
Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of
images and train a classifier Linear SVM classifier

Optionally, you can also apply a color transform and append binned color features, as well as
histograms of color, to your HOG feature vector.

Note: for those first two steps don't forget to normalize your features and randomize a selection
for training and testing.

Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
    
Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4)
and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.

Estimate a bounding box for vehicles detected.

extract hog for the whole images at once
normalize training data
randomly shuffling the data
'''

################
#1. read in all images
################

path = '../data/'

car = []
noncar = []

car_dir = glob.glob(path+'vehicles/*')
noncar_dir = glob.glob(path+'non-vehicles/*')

#read in all images
test = glob.glob(path+'test_images/*.jpg')
for dir in car_dir:
    car = car + glob.glob(dir+'/*.png')
for dir in noncar_dir:
    noncar= noncar + glob.glob(dir+'/*.png')

#define the label
y_car = np.ones(len(car))
y_noncar = np.zeros(len(noncar))

################
#2. extract feature
################

def get_hog_features():
    pass

def get_hog_features_u(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    if vis == True:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False,
                                  visualise=True, feature_vector=False)
        return features, hog_image
    else:
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False,
                       visualise=False, feature_vector=feature_vec)
        return features
