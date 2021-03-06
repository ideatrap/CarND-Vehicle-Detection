import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
import pickle
import time
from scipy.ndimage.measurements import label
import scipy.misc


################
#1. read in all images
################
path = '../data/'
path_write ='../data/vehicles/KITTI_extracted/92.png'
#path_write= '../data/non-vehicles/Extras/extra34.png'

def import_image():

    #read in all images
    car = glob.glob(path+'vehicles/*/*.png')
    noncar= glob.glob(path+'non-vehicles/*/*.png')

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


'''
#Visualize HOG feature
img = cv2.imread(path_write)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
img = cv2.resize(img, (64,64))

features, hog_image = get_hog_features(img[:,:,0], orient, pix_per_cell, cell_per_block, vis=True)
plt.imshow(hog_image)
plt.show()
'''


def extract_features(imgs, cspace,
                     orient, pix_per_cell, cell_per_block, hog_channel,
                     spatial_size=(32, 32), hist_bins=32, hist_range=(0, 256)):
    features = []
    for file in imgs:
        # Read in each one by one
        image = cv2.imread(file)
        # apply color conversion if other than 'BGR'

        image = cv2.resize(image, (64,64))
        if cspace != 'BGR':
            if cspace == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            elif cspace == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2LUV)
            elif cspace == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
            elif cspace == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
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

        # Apply bin_spatial() to get spatial color features
        spatial_features = bin_spatial(feature_image, size=spatial_size)

        # Apply color_hist() to get color histogram features
        hist_features = color_hist(feature_image, nbins=hist_bins, bins_range=hist_range)


        # Append feature vector to the features list
        features.append(np.hstack((hog_features, spatial_features, hist_features)))
        #features.append(hog_features)

    # Return list of feature vectors
    return features


def feature_extraction_process(car, noncar, colorspace, orient, pix_per_cell, cell_per_block, hog_channel):
    t = time.time()
    X_car = extract_features(car, cspace=colorspace, orient=orient,
                             pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel)
    X_noncar = extract_features(noncar, cspace=colorspace, orient=orient,
                             pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel)

    t2 = time.time()
    print(round(t2 - t, 2), 'Seconds to extract features.')

    X = np.vstack((X_car, X_noncar)).astype(np.float64)

    if len(X)>0:
        global scaler
        scaler = StandardScaler().fit(X)
        scaled_X = scaler.transform(X)


    #define label vetor
    y = np.hstack((np.ones(len(X_car)), np.zeros(len(X_noncar))))

    #store the result as pickle
    with open('scaler.pickle', 'wb') as file:
        pickle.dump(scaler, file)

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

    with open(path + 'y.pickle', 'rb') as file:
        y = pickle.load(file)

    with open(path+'scaled_x.pickle', 'rb') as file:
        scaled_X = pickle.load(file)

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

    with open('svc.pickle', 'wb') as file:
        pickle.dump(svc, file)
    print('Model saved.')

    n_predict = 10
    print('My SVC predicts: \n', svc.predict(X_test[0:n_predict]))
    print('For these', n_predict, 'labels: \n', y_test[0:n_predict])
    t2 = time.time()


################
# 4. sliding window
################

def draw_boxes(img, bboxes, color=(255, 0, 0), thick=4):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates

        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy

#simple slide window function - slide through the image
def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None],
                 xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    #xy_window defines the size of the search area in number of pixels
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
    ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
    nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step)
    ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step)
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list


def convert_color(img, colorspace='YUV'):
    if colorspace == 'YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    elif colorspace == 'LUV':
        return cv2.cvtColor(img, cv2.COLOR_BGR2LUV)
    elif colorspace == 'YUV':
        return cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    elif colorspace == 'RGB':
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        print('Color conversion error! Color code not recognized')
        exit()


#HOG sub-sampling to slide through the image
def sub_sampling(img, ystart, ystop, scale, svc, X_scaler, colorspace, orient,
                pix_per_cell, cell_per_block, spatial_size = (32, 32), hist_bins = 32):

   #img = img.astype(np.float32) / 255
   window_list= []
   img_tosearch = img[ystart:ystop, :, :]
   ctrans_tosearch = convert_color(img_tosearch, colorspace=colorspace)


   if scale != 1:
       imshape = ctrans_tosearch.shape
       ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))

   ch1 = ctrans_tosearch[:, :, 0]
   ch2 = ctrans_tosearch[:, :, 1]
   ch3 = ctrans_tosearch[:, :, 2]


   # Define blocks and steps as above
   nxblocks = (ch1.shape[1] // pix_per_cell) -1
   nyblocks = (ch1.shape[0] // pix_per_cell) - 1
   nfeat_per_block = orient * cell_per_block ** 2

   window = 64 #8X8, 8 pixel per cell, 8 cells per block
   nblocks_per_window = (window // pix_per_cell) -1
   cells_per_step = 2  # skip 62.5% every move
   nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
   nysteps = (nyblocks - nblocks_per_window) // cells_per_step

   # Compute individual channel HOG features for the entire image
   hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, vis=False,feature_vec=False)
   hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, vis=False,feature_vec=False)
   hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, vis=False,feature_vec=False)


   for xb in range(nxsteps):
       for yb in range(nysteps):
           ypos = yb * cells_per_step
           xpos = xb * cells_per_step


           # Extract HOG for this patch
           hog_feat1 = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
           hog_feat2 = hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
           hog_feat3 = hog3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()


           hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

           xleft = xpos * pix_per_cell
           ytop = ypos * pix_per_cell

           # Extract the image patch
           subimg = cv2.resize(ctrans_tosearch[ytop:ytop + window, xleft:xleft + window], (64, 64))

           # Get color features
           spatial_features = bin_spatial(subimg, size=spatial_size)

           hist_features = color_hist(subimg, nbins=hist_bins, bins_range=(0, 256))

           # Scale features and make a prediction
           features = X_scaler.transform(np.hstack((hog_features, spatial_features, hist_features)).reshape(1,-1))
           #features = X_scaler.transform(hog_features.reshape(1,-1))
           #features = hog_features

           features = np.array(features)
           prediction = svc.predict(features)

           if prediction == 1:
               xbox_left = np.int(xleft * scale)
               ytop_draw = np.int(ytop * scale)
               win_draw = np.int(window * scale)
               window_list.append(((int(xbox_left), int(ytop_draw + ystart)),
                                   (int(xbox_left + win_draw), int(ytop_draw + win_draw + ystart))))

   return window_list



def find_car_window(img, colorspace, orient, pix_per_cell, cell_per_block, start_stop_scale,
                    spatial_size = (32, 32), hist_bins = 32):
    windows = []

    with open('svc.pickle', 'rb') as file:
        svc = pickle.load(file)

    with open('scaler.pickle', 'rb') as file:
        scaler = pickle.load(file)

    ystart_ls= start_stop_scale[0]
    ystop_ls = start_stop_scale[1]
    scale_ls = start_stop_scale[2]


    for ystart, ystop, scale in zip(ystart_ls, ystop_ls, scale_ls):

        window = sub_sampling(img, ystart, ystop, scale, svc, scaler, colorspace, orient,
                              pix_per_cell, cell_per_block,
                              spatial_size= spatial_size, hist_bins= hist_bins)

        windows = windows + window
    return windows

################
#5. Heatmap
################
def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    return heatmap

def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image

        cv2.rectangle(img, bbox[0], bbox[1], (255, 0, 0), 4)
    # Return the image
    return img

#test the whole pipeline on single image
def test_pipeline(path):
    img = cv2.imread(path)
    windows = find_car_window(img, colorspace, orient, pix_per_cell, cell_per_block, start_stop_scale)



    if len(windows) > 0:

        heat = np.zeros_like(img[:, :, 0]).astype(np.float)
        heat = add_heat(heat, windows)
        heat = apply_threshold(heat, 2)  # remove false positive

        heatmap = np.clip(heat, 0, 255)  # make it 3 channel image
        labels = label(heatmap)
        draw_img = draw_labeled_bboxes(np.copy(img), labels)
        draw_img = cv2.cvtColor(draw_img, cv2.COLOR_BGR2RGB)

        #Draw the individual window
        #draw_img = draw_boxes(draw_img, windows)
        #plt.imshow(draw_img)

        #display the heat map and marked image
        fig = plt.figure()
        plt.subplot(211)
        plt.imshow(draw_img)
        plt.subplot(212)
        plt.imshow(heatmap, cmap='hot')
        fig.tight_layout()

        plt.show()


    else:
        print('No car identified')


################
#6. Video Editing
################

index = 0
def pipeline(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # find all windows containing car
    global index


    path_image_video_o = '../data/original/' + str(index) +'.jpg'
    path_image_video_marked = '../data/marked/' + str(index) + '.jpg'

    #to save the original image
    #scipy.misc.imsave(path_image_video_o, img)

    windows = find_car_window(img, colorspace, orient, pix_per_cell, cell_per_block, start_stop_scale)

    if len(windows) > 0:
        heat = np.zeros_like(img[:, :, 0]).astype(np.float)
        heat = add_heat(heat, windows)
        heat = apply_threshold(heat, 2)  # remove false positive

        heatmap = np.clip(heat, 0, 255)  # make it 3 channel image
        labels = label(heatmap)
        draw_img = draw_labeled_bboxes(np.copy(img), labels)

    else:
        draw_img = np.copy(img)

    #scipy.misc.imsave(path_image_video_marked, draw_img)
    draw_img = cv2.cvtColor(draw_img, cv2.COLOR_RGB2BGR)
    index += 1
    return draw_img


from moviepy.editor import VideoFileClip

def process_video(video_path):
    #clip1 = VideoFileClip(video_path).subclip(41,43)
    clip1 = VideoFileClip(video_path)
    video = clip1.fl_image(pipeline)
    video.write_videofile('output.mp4', audio=False)


################
#full pipeline
################

#1 import images
#car, noncar = import_image()

colorspace = 'YUV'
orient = 11
pix_per_cell = 8
cell_per_block = 2
hog_channel = 'ALL'

#2.extract feature from images
#feature_extraction_process(car, noncar, colorspace, orient, pix_per_cell, cell_per_block, hog_channel)

#3. train SVM model
#train_model()

#4. test the pipeline to slide window to draw box for single image

ystart_ls = [380,405,400,410,400]
ystop_ls = [540,550,600,630,800]
scale_ls = [1.2, 1.2,1.4,1.4, 1.7]


start_stop_scale = np.vstack((ystart_ls,ystop_ls, scale_ls))


path_test ='../data/test_images/test5.jpg'
test_pipeline(path_test)


#6 process the video
path_video =path+"project_video.mp4"
#path_video =path+"test_video.mp4"
#process_video(path_video)

