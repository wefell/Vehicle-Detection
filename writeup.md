##Writeup for Project 5

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.


## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this is located in the second cell of vehicle_detection.ipynb, and was taken from the lectures (All of the lesson functions are located in cell 2. The find_cars function in cell 2 has been slightly modified to return windows instead of a heatmap, and to accept a list of scales).

```
def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, 
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), 
                                  transform_sqrt=False, 
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:      
        features = hog(img, orientations=orient, 
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), 
                       transform_sqrt=False, 
                       visualise=vis, feature_vector=feature_vec)
        return features
```

I started by exploring the different color spaces in 3D plots to get an idea for the how the features separate and how consistent the difference is between cars and non-cars. HSV seemed to give the best accuracy at first, but YCrCb, YUV, and LUV interestingly seemed to lead to less false positives. I ended up using YCrCb after a lot of trial and error testing on test images and videos. Non-car images seem to look very similar to other non-car images in the  YCrCb 3D plot, and the same is true of car images compared to other car images. I believe this helps with generalization. HSV does usually provide clear, and sometimes drastic differences between car and non-car images, but patterns between images of the same class can differ quite a bit. I see how HSV could lead to more overfitting compared to YCrCb.

Car
![alt text](https://github.com/wefell/Vehicle-Detection/blob/master/ouput_images/car.png "Car")
Non Car
![alt text](https://github.com/wefell/Vehicle-Detection/blob/master/ouput_images/noncar.png "Non Car")
![alt text](https://github.com/wefell/Vehicle-Detection/blob/master/ouput_images/rgb_car.png "RGB Car")
![alt text](https://github.com/wefell/Vehicle-Detection/blob/master/ouput_images/rgb_noncar.png "RGB Non Car")
![alt text](https://github.com/wefell/Vehicle-Detection/blob/master/ouput_images/hsv_car.png "HSV Car")
![alt text](https://github.com/wefell/Vehicle-Detection/blob/master/ouput_images/hsv_noncar.png "HSV Non Car")
![alt text](https://github.com/wefell/Vehicle-Detection/blob/master/ouput_images/luv_car.png "LUV Car")
![alt text](https://github.com/wefell/Vehicle-Detection/blob/master/ouput_images/luv_noncar.png "LUV Non Car")
![alt text](https://github.com/wefell/Vehicle-Detection/blob/master/ouput_images/hls_car.png "HLS Car")
![alt text](https://github.com/wefell/Vehicle-Detection/blob/master/ouput_images/hls_noncar.png "HLS Non Car")
![alt text](https://github.com/wefell/Vehicle-Detection/blob/master/ouput_images/yuv_car.png "YUV Car")
![alt text](https://github.com/wefell/Vehicle-Detection/blob/master/ouput_images/yuv_noncar.png "YUV Non Car")
![alt text](https://github.com/wefell/Vehicle-Detection/blob/master/ouput_images/YCrCb_car.png "YCrCb Car")
![alt text](https://github.com/wefell/Vehicle-Detection/blob/master/ouput_images/YCrCb_noncar.png "YCrCb Non Car")


I also grabbed random images and explored the different HOG channels. I decided to use all three HOG channels, to gain the most feature imformation.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image2]

####2. Explain how you settled on your final choice of HOG parameters.

My final choice of HOG parameters were chosen through a long experimentation process with test images and videos. The goal was to place as many windows on cars as possible, with as little false positives as possible. A second consideration in selecting parameters was speed at which features can be extracted. 

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

The code to train my classifier is displayed below. Again, my final choice of parameters were settled after much trail and error in reducing false positives and false negatives. After much experimentation with parameters I then found myself experimenting with the data. I found some partial images of cars in the non-car data, so I removed those. This did not seem to help with decreasing false positives. I also augmented the dataset and flipped all of the images, but this did not seem to help either. I eventually went back to the original, untouched data set and started experimenting with the classifier itself. I used the LinearSCV classifier, and bumped the parameter C first up to 1.0e5 and then down to 1.0e-5, and then up and down that range. I found a parameter C setting of 1.0e-5 to give the best performance in reducing false positives and false negatives. I also experimented with an SVC classifier with an rbf kernal and a randomforest kernal, and found them to give very exciting results. I ended up abondoning those two classifiers for lack of time, as they take incredibly long to train.

```
# Define feature parameters
color_space = 'YCrCb'  # RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9
pix_per_cell = 8
cell_per_block = 2
hog_channel = 'ALL'  # 0, 1, 2, "ALL"
spatial_size = (32, 32)
hist_bins = 32
spatial_feat = True
hist_feat = True
hog_feat = True

t = time.time()

car_features = extract_features(
            cars, color_space=color_space, spatial_size=spatial_size,
            hist_bins=hist_bins, orient=orient, pix_per_cell=pix_per_cell,
            cell_per_block=cell_per_block, hog_channel=hog_channel,
            spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)

non_car_features = extract_features(
            non_cars, color_space=color_space, spatial_size=spatial_size,
            hist_bins=hist_bins, orient=orient, pix_per_cell=pix_per_cell,
            cell_per_block=cell_per_block, hog_channel=hog_channel,
            spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)

print(round(time.time()-t, 2), 'Seconds to extract features...')

X = np.vstack((car_features, non_car_features)).astype(np.float64)
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X)

# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(non_car_features))))

# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
    scaled_X, y, test_size=0.10, random_state=rand_state)

print('Using: ', orient, 'orientations, ', pix_per_cell,
      'pixels per cell, ', cell_per_block, 'cells per block, ',
      hist_bins, 'histogram bins, and ', spatial_size, 'spatial sampling')
print('Feature vector length: ', len(X_train[0]))

# Linear SVC
# svc = SVC(kernel='rbf')
svc= LinearSVC(C=0.00001)
# from sklearn.ensemble import RandomForestClassifier
# svc = RandomForestClassifier(n_estimators=5, min_samples_leaf=2)
t = time.time()
svc.fit(X_train, y_train)

print(round(time.time()-t, 2), 'Seconds to train SVC...)')
# Check accuracy
print('Test Accuracy: ', round(svc.score(X_test, y_test), 4))
```

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

This code was also taken from the lessons. As with most parameters, I decided on my scales by experimentall reducing the number of false positive and false negatives. More than two scales seemed to be a needless waste of time. One scale of 1.5 works reasonably well, yet two scales of 1.5 an 2.0 seem to be lead to less moments where the cars are not identified.

```
searchpath = 'test_images/*'
test_images = glob.glob(searchpath)
heat = []
heat_titles = []
final = []
y_start_stop = (400, 650)
scales = [1.5, 2]

for img in test_images:
    img = cv2.imread(img)
    output_img, windows = find_cars(img, y_start_stop[0], y_start_stop[1], scales, svc, X_scaler, orient,
                                    pix_per_cell, cell_per_block, spatial_size, hist_bins)
    
    heat_map = np.zeros_like(img[:, :, 0])
    heat_map = add_heat(heat_map, windows)
    heat_map = apply_threshold(heat_map, 0)
    heat_map = np.clip(heat_map, 0, 255)
    
    heat.append(output_img)
    heat.append(heat_map)
    heat_titles.append(' ')
    heat_titles.append(' ')
        
    labels = label(heat_map)
    draw_img = draw_labeled_bboxes(np.copy(img), labels)
    final.append(draw_img)
    final.append(heat_map)
    
fig = plt.figure(figsize=(12, 18), dpi=100)
visualize(fig, 8, 2, heat, heat_titles)

fig = plt.figure(figsize=(12, 18), dpi=100)
visualize(fig, 8, 2, final, heat_titles)
```

![alt text][image3]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

See above for explanation of what I did to optimize my classifier itself. Below are test images comparing the differences in parameter C settings.

![alt text][image4]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections over 15 frames I created a heatmap and then thresholded that map to identify only vehicle positions that appeared in 9 out of the last 15 frames.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

My implimentation still has a few false positives, but is much better after using several frames to form a heatmap, applying a threshold, and opimizing the parameter C in my LinearSVC classifer. If I had much more time, I would experiment more with augmented data and using different classifiers such as SVC with rbf kernal or a random forest classifier.


