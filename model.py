import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
import pickle
import time

'''

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
def import_image():

    car = []
    noncar = []

    car_dir = glob.glob(path+'vehicles/*')
    noncar_dir = glob.glob(path+'non-vehicles/*')

    #read in all images

    for dir in car_dir:
        car = car + glob.glob(dir+'/*.png')
    for dir in noncar_dir:
        noncar= noncar + glob.glob(dir+'/*.png')

    print('{} car images imported.'.format(len(car)))
    print('{} non-car images imported.'.format(len(noncar)))

    return car, noncar

################
#2. extract features
################

# Define a function to compute binned color features
def bin_spatial(img, size=(32, 32)):
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel()
    # Return the feature vector
    return features


def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
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


def extract_features(imgs, cspace='RGB',
                     orient=9, pix_per_cell=8, cell_per_block=2, hog_channel=0,
                     spatial_size=(32, 32), hist_bins=32, hist_range=(0, 256)):
    features = []
    for file in imgs:
        # Read in each one by one
        image = mpimg.imread(file)
        # apply color conversion if other than 'RGB'
        if cspace != 'RGB':
            if cspace == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif cspace == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif cspace == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif cspace == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        else:
            feature_image = np.copy(image)

        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.append(get_hog_features(feature_image[:, :, channel],
                                                     orient, pix_per_cell, cell_per_block,
                                                     vis=False, feature_vec=True))
            hog_features = np.ravel(hog_features)
        else:
            hog_features = get_hog_features(feature_image[:, :, hog_channel], orient,
                                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)

        # Append the new feature vector to the features list
        features.append(hog_features)

        # Apply bin_spatial() to get spatial color features
        #spatial_features = bin_spatial(feature_image, size=spatial_size)

        # Apply color_hist() to get color histogram features
        # hist_features = color_hist(feature_image, nbins=hist_bins, bins_range=hist_range)

        # Append the new feature vector to the features list
        #features.append(np.concatenate((spatial_features, hist_features)))

    # Return list of feature vectors
    return features

#normalize the input
def normalize(feature):
    if len(feature)>0:
        scaler = StandardScaler().fit(feature)
        scaled_feature = scaler.transform(feature)
        return scaled_feature
    else:
        print('Error! Normalization function received empty feature list.')
        exit()


def feature_extraction_process(car, noncar):

    colorspace = 'YUV'  #based on test, HLS appears a good choice
    orient = 11
    pix_per_cell = 8
    cell_per_block = 3
    hog_channel = 1  # Can be 0, 1, 2, or "ALL"

    t = time.time()
    X_car = extract_features(car, cspace=colorspace, orient=orient,
                             pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel)
    X_noncar = extract_features(noncar, cspace=colorspace, orient=orient,
                             pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel)

    t2 = time.time()
    print(round(t2 - t, 2), 'Seconds to extract HOG features...')
    X = np.vstack((X_car, X_noncar)).astype(np.float64)
    scaled_X = normalize(X)

    #define label vetor
    y = np.hstack((np.ones(len(X_car)), np.zeros(len(X_noncar))))

    #store the result as pickle
    with open(path+'scaled_x.pickle', 'wb') as file:
        pickle.dump(scaled_X, file)

    with open(path+'y.pickle', 'wb') as file:
        pickle.dump(y, file)

    print('Feature extraction results are stored in pickle.')


################
# 3. train SVM model
################
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC


def train_model():
    with open(path+'scaled_x.pickle', 'rb') as file:
        scaled_X= pickle.load(file)

    with open(path + 'y.pickle', 'rb') as file:
        y = pickle.load(file)

    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.22, random_state=rand_state)

    svc = LinearSVC()

    # Check the training time for the SVC
    t = time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2 - t, 2), 'seconds to train SVC...')
    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))

    with open(path+'svc.pickle', 'wb') as file:
        pickle.dump(svc, file)
    print('Model saved.')

################
#full pipeline
################

#1 import images
#car, noncar = import_image()

#2.extract feature from images
#feature_extraction_process(car, noncar)

#3. train SVM model
#train_model()


