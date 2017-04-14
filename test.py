import glob
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2


#path = '../data/non-vehicles/Extras/extra10.png'
path ='../data/vehicles/KITTI_extracted/60.png'
image = []
image.append(mpimg.imread(path))

#image.append(cv2.cvtColor(image[0], cv2.COLOR_RGB2HSV))
#image.append(cv2.cvtColor(image[0], cv2.COLOR_RGB2LUV))
image.append(cv2.cvtColor(image[0], cv2.COLOR_RGB2HLS))
image.append(cv2.cvtColor(image[0], cv2.COLOR_RGB2YUV))
#image.append(cv2.cvtColor(image[0], cv2.COLOR_RGB2YCR_CB))
#image.append(cv2.cvtColor(image[0], cv2.COLOR_RGB2LAB))


for item in image:
    plt.imshow(item)
    plt.show()
