import glob
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2

from model import *

path2 = '../data/non-vehicles/Extras/extra10.png'
#path2 ='../data/vehicles/KITTI_extracted/92.png'
#path2= '../data/non-vehicles/Extras/extra34.png'
image = []
image.append(cv2.imread(path2))

#image.append(cv2.cvtColor(image[0], cv2.COLOR_RGB2HSV))
#image.append(cv2.cvtColor(image[0], cv2.COLOR_RGB2LUV))
image.append(cv2.cvtColor(image[0], cv2.COLOR_RGB2HLS))
image.append(cv2.cvtColor(image[0], cv2.COLOR_RGB2YUV))
#image.append(cv2.cvtColor(image[0], cv2.COLOR_RGB2YCR_CB))
#image.append(cv2.cvtColor(image[0], cv2.COLOR_RGB2LAB))



with open('scaler.pickle', 'rb') as file:
    scaler = pickle.load(file)



feature = extract_features([path2], cspace=colorspace, orient=orient,
                           pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel)

feature = scaler.transform(feature)

with open('feature.pickle', 'wb') as file:
    pickle.dump(feature, file)
'''
# test the prediction capability --> it's very robust
with open('svc.pickle', 'rb') as file:
    svc = pickle.load(file)

with open('scaler.pickle', 'rb') as file:
    scaler = pickle.load(file)
feature = scaler.transform(feature)

'''

print(len(feature[0]))
#print(svc.predict(feature))