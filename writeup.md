## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./test_images_output/1.2chessboard_undistort_output.jpg "Undistorted"
[image9]: ./test_images_output/2.1image_undistort_output.jpg "Original image undistorted"
[image2]: ./test_images_output/2.2binary_output_hsl_maxtrix.jpg "HSL matrix, Binary"
[image3]: ./test_images_output/2.2binary_output.jpg "Binary Example"
[image4]: ./test_images_output/2.2binary_output_masking.jpg "Binary Example after masking"
[image5]: ./test_images_output/2.3perspectivetransform_output.jpg "Pespective Transformation"
[image6]: ./test_images_output/2.4perspectivetransform_boxandline.jpg "espective Transformation with box"
[image7]: ./test_images_output/2.5original_curvature_offset.jpg "Original image with curvature info"
[image8]: ./test_images_output/2.6original_highlightedlane.jpg "Original image with highlighted lane"
[video1]: .test_videos_output/project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the first code cell of the IPython notebook located in **1.Camera Calibration** of "./P2.ipynb".

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result:

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image9]

Use `cv2.undistort()` function to obtain picture above.

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image in **2.2. Binary Picture Filtering & Masking**.  Here's an example of my output for this step.

![alt text][image3]

Matrix below showes different binary picture by applying different filter. According to this, **gradx,s** and **gradx,l** are selected to generate binary picture.
![alt text][image2]

Area of interested is applied to filtering out unnecesary area. Coordiante as below:
```IPython
left_bottom = [150, 720]
left_upper = [550, 400]
right_upper = [780,400]
right_bottom = [1200, 720]
```
![alt text][image4]
#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.
![alt text][image5]

The code for my perspective transform includes a function called `img_unwarp()`, which appears in the **Helper Functions** section.  The `img_unwarp()` function takes as inputs an image (`img`), distortion matrix (`mtx`),distortion factor (`dist`).In addition, the source (`src`) and destination (`dst`) points is hardcoded inside the function:
```python
src = np.float32([[223,719], # lower left corner
                  [582,456], # upper left corner
                  [694,456], # upper right corner
                  [1087,719]]) # lower right corner

dst = np.float32([[320,720], # lower left corner
                  [320,0], # upper left corner
                  [960,0], # upper right corner
                  [960,720]]) # lower right corner
```

This resulted in the following source and destination points:

| Source        | Destination   |
|:-------------:|:-------------:|
| 223, 719      | 320, 720      |
| 582, 456      | 320, 0        |
| 694, 456      | 960, 0        |
| 1087, 719     | 960, 720      |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.



#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

The code to fit my lane lines with a 2nd order polynomial includes a function called `fit_polynomial()`, which appears in the **Helper Functions** section.  The `fit_polynomial()` function takes as input an image (`img`) after perspective transformation. Result is shown below:

![alt text][image6]

The location of the 1st window is shown based on 1/3 part of the bottom picture. Then the 2nd window is chosen based on mean of the pixel inside the window area.

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.
![alt text][image7]

I did this by using function `measure_curvature_real()` and `offset_x`. Both functions need to take the conversion ratio as below:

```IPython
xm_per_pix = 3.7/640 # meters per pixel in x dimension
ym_per_pix = 12/720 # meters per pixel in y dimension
```
The ym_per_pix is estimated from the dimension from [road marking](https://www.civil.iitb.ac.in/~vmtom/1100_LnTse/525_lnTse/plain/), where:    
    dash = 1.5 meters\
    gap = 3.0 meters


#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

The code to overlap the lane to the original image is called `lane_highlight()`, which appears in the **Helper Functions** section. The `ane_highlight()` function would take original image (`img`), binary top down view image (`top_down`), invert transform matrix (`perspective_Minv`), and the point on the fited line (`polyfit_pts`). Result is shown below:

![alt text][image8]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

**Discussion 1: Imaging Filter & Masking**\
More work could be done to improve the binary image to reduce noise **1)** from unwanted object **2)** from shadow **3)** from other line on the road by amending

**Discussion 2: Perspective Transfrom**\
Some pictures doesn't work well in current perspective transformation. 
