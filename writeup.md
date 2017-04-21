
**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.


---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  

This is the writeup.

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the section `2. extract features` of `model.py`.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

Car:

[![image0011.png](https://s25.postimg.org/iz0pkn27z/image0011.png)](https://postimg.org/image/5i3r1rrwb/)

Non-car:

[![image7.png](https://s25.postimg.org/5gtt8cq2n/image7.png)](https://postimg.org/image/s5j07x7gb/)

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).

Here is an example using the `YUV` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

Car Image:

![Car](https://s25.postimg.org/42xkzb1fj/car.png)

Car with YUV:

[![Screen Shot 2017-04-21 at 11.01.19 PM.png](https://s25.postimg.org/r87zlb6kf/Screen_Shot_2017-04-21_at_11.01.19_PM.png)](https://postimg.org/image/6o35mtqt7/)

YUV HOG Features:

[![Screen Shot 2017-04-21 at 11.10.08 PM.png](https://s25.postimg.org/o2ndv3ny7/Screen_Shot_2017-04-21_at_11.10.08_PM.png)](https://postimg.org/image/cdje74wzf/)



#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combination of `orientations`, `pixels_per_cell` and `cells_per_block`. I found that increasing `orientations`from 8 to 11, increasing the prediction accuracy. But increasing `pix_per_cell` doesn't help with accuracy, and I kept it at 8.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using three feature: `HOG`, `Color Spatial` and `Color Histogram`. I compared the prediction accuracy with case of using only one of the feature. The result turns out using three features together gives better accuracy.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The slide search is implemented in `section 4`, in function `sub_sampling()`.

I scan through the image with sliding widows. The sliding window is set with parameter of `ystart` (start point from y dimension), `yend`(end point on y dimension) and `scale` (multiplier on 8 by 8 square sliding window).

I set the `ystart` to be the middle of the image, where cars starts to appear. This helps to reduce false positive, and improve performance.

I found that that for `scale` less than 1 dramatically increases the false positive, and anything bigger than 2 captures region usually too big. `scale` around 1.5 is most useful, and it captures the full car image.

I also found that the `ystar` also affects effectiveness. slight change on `ystart` will affect whether the window can capture the car image. Therefore, since the car keeps changing position, I use the same `scale` for different `ystart`.

I also keep every step movement to be 25% of the window size, so that the whole region is properly scanned.

[![Screen Shot 2017-04-21 at 11.43.31 PM.png](https://s25.postimg.org/xpvhik6bj/Screen_Shot_2017-04-21_at_11.43.31_PM.png)](https://postimg.org/image/watwtu58b/)

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched using YUV channel with combined HOG, spatially binned color and histograms of color.  Here are some example images:

[![Screen Shot 2017-04-22 at 12.14.59 AM.png](https://s25.postimg.org/soo3h6yv3/Screen_Shot_2017-04-22_at_12.14.59_AM.png)](https://postimg.org/image/mnqek4c8r/)

[![Screen Shot 2017-04-22 at 12.16.06 AM.png](https://s25.postimg.org/85xspvabj/Screen_Shot_2017-04-22_at_12.16.06_AM.png)](https://postimg.org/image/9xqrkrtob/)

[![Screen Shot 2017-04-22 at 12.17.07 AM.png](https://s25.postimg.org/rp2dz8933/Screen_Shot_2017-04-22_at_12.17.07_AM.png)](https://postimg.org/image/hezyzzj7f/)
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](https://youtu.be/9chFnWx5I2E)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I ran the script on the full video, and identified the subclip false positive happens. I then used `scipy.misc.imsave()`to save every frame of the sub clip and adjust the parameters to improve sensitivity and minimize false positive.


I also used heatmap and then thresholded to filter positive and identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here are some examples:

[![Screen Shot 2017-04-22 at 12.27.56 AM.png](https://s25.postimg.org/vmpnomvwf/Screen_Shot_2017-04-22_at_12.27.56_AM.png)](https://postimg.org/image/nh7lqh7nf/)[![Screen Shot 2017-04-22 at 12.33.06 AM.png](https://s25.postimg.org/6hynb7wfz/Screen_Shot_2017-04-22_at_12.33.06_AM.png)](https://postimg.org/image/m3fyv68e3/)



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Challenge:
  To get a good outcome, the model training is easiest part, but the data pipeline to feed in the right data is super challenging.

  1) **Color conversion**. I read in image from `cv2.imread()` in `BGR` format, I had carefully converted image for display. However, `VideoFileClip.fl_image()` reads in image in `RGB` format, and makes the prediction slightly off.

  2) **normalization**. As I was switching between `cv2.imread()` and `mpimg.imread()`, I had one liner `img = img.astype(np.float32) / 255` to scale the image, and I messed it up. In the end, my model doesn't work, and always predict positive on all `.jpg` images no matter what is fed in. I only discovered it through forum.

Improvement:
Right now, the model still captures the car on the opposite side road as positive. It's a right thing to do because it's indeed a car, and it should detect it in case the car runs on the wrong side of the road.

However, to properly act on this information, I shall add a filter to track the movement direction and position of the car, so that I can take proper action. Additional, I should train the model to recognize whether it's the face or tail of the car, so that I can separate the cases.
