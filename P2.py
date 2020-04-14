#!/usr/bin/env python
# coding: utf-8

# # Self-Driving Car Engineer Nanodegree
# 
# 
# ## Advanced Lane Finding Project
# 
# The goals / steps of this project are the following:
# 
# * Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
# * Apply a distortion correction to raw images.
# * Use color transforms, gradients, etc., to create a thresholded binary image.
# * Apply a perspective transform to rectify binary image ("birds-eye view").
# * Detect lane pixels and fit to find the lane boundary.
# * Determine the curvature of the lane and vehicle position with respect to center.
# * Warp the detected lane boundaries back onto the original image.
# * Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.
# 
# ---

# ## Helper Functions
# Below are some helper functions to help get you started. They should look familiar from the lesson!
# 
# 
# **Run the cell below to import some packages.  If you get an `import error` for a package you've already installed, try changing your kernel (select the Kernel menu above --> Change Kernel).  Still have problems?  Try relaunching Jupyter Notebook from the terminal prompt.  Also, consult the forums for more troubleshooting tips.**  

# In[22]:


import math
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
get_ipython().run_line_magic('matplotlib', 'qt')
#%matplotlib inline

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image
    

def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, γ)

def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    
    # Apply the following steps to img
    # 1) Convert to saturation scale
    hls = cv2.cvtColor(img,cv2.COLOR_RGB2HLS)
    sat = hls[:,:,2] # extract satuarion for yellow
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    if orient == 'x':
        sobel = cv2.Sobel(sat,cv2.CV_64F,1,0,ksize=sobel_kernel)
    elif orient == 'y':
        sobel = cv2.Sobel(sat,cv2.CV_64F,0,1,ksize=sobel_kernel)
    # 3) Take the absolute value of the derivative or gradient
    abs_sobel = np.absolute(sobel)
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scale_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # 5) Create a mask of 1's where the scaled gradient magnitude 
            # is > thresh_min and < thresh_max
    binary_output = np.zeros_like(sat)
    binary_output[(scale_sobel > thresh[0]) & (scale_sobel < thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    return binary_output

def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    
    # Apply the following steps to img
    # 1) Convert to saturation scale
    hls = cv2.cvtColor(img,cv2.COLOR_RGB2HLS)
    sat = hls[:,:,2] # extract satuarion for yellow
    # 2) Take the gradient in x and y separately
    sobel_x = cv2.Sobel(sat,cv2.CV_64F,1,0,ksize=sobel_kernel)
    sobel_y = cv2.Sobel(sat,cv2.CV_64F,0,1,ksize=sobel_kernel)
    # 3) Calculate the magnitude 
    sobel_mag = np.sqrt(np.power(sobel_x,2)+np.power(sobel_y,2))
    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scale_sobel = np.uint8(255*sobel_mag/np.max(sobel_mag))
    # 5) Create a binary mask where mag thresholds are met
    binary_output = np.zeros_like(sat)
    binary_output[(scale_sobel > mag_thresh[0]) & (scale_sobel < mag_thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    return binary_output

def color_select(img,gray_threshold):
    """
    Find the white color inside the image
    """
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    
    # 2) Create a binary mask where mag thresholds are met
    binary_output = np.zeros_like(gray)
    color_ind = (gray>gray_threshold)
    binary_output[color_ind] = 1
    # 6) Return this mask as your binary_output image
    return binary_output

def img_unwarp(img, mtx, dist):
    """
    This function will undistrot the image and gernerate top-down view of image
    """
    undist_img = cv2.undistort(img, mtx, dist, None, mtx)

    # 1) Measure the source point and it's destination couterpart. 
    src = np.float32([[223,719], # lower left corner
                      [582,456], # upper left corner 
                      [694,456], # upper right corner
                      [1087,719]]) # lower right corner

    dst = np.float32([[320,720], # lower left corner
                      [320,0], # upper left corner 
                      [960,0], # upper right corner
                      [960,720]]) # lower right corner
    # 2) use cv2.getPerspectiveTransform() to get M, the transform matrix
    M = cv2.getPerspectiveTransform(src,dst)
    Minv = cv2.getPerspectiveTransform(dst,src)

    # 3) use cv2.warpPerspective() to warp your image to a top-down view
    img_size = (undist_img.shape[1],undist_img.shape[0])
    warped = cv2.warpPerspective(undist_img,M,img_size,flags=cv2.INTER_LINEAR)
    return warped, M, Minv, src


def find_lane_pixels(binary_warped):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    # Create an output image to draw on and visualize the result. Need to have 3 channels
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    # leftx_base & rightx_base is based on bottom half of the image (NOT SLIDING WINDOWS)
    leftx_base = np.argmax(histogram[:midpoint])
    # The reason to add '+ midpoint' is that rightx_base need to pass the midpoint
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    'Why do we need this????'
    minpix = 50

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0]//nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero() # Return the indices of non-zero value, in 2 1-D-arrays having same length
    nonzeroy = np.array(nonzero[0]) # The first array is y-dimension
    nonzerox = np.array(nonzero[1]) # The second array is x-dimension
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        ### TO-DO: Find the four below boundaries of the window ###
        win_xleft_low = leftx_current-margin  # left window, lower x
        win_xleft_high = leftx_current+margin  # left window, higher x
        win_xright_low = rightx_current-margin  # right window, lower x
        win_xright_high = rightx_current+margin  # right window, higher x
        
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),
        (win_xleft_high,win_y_high),(0,255,0), thickness=2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),
        (win_xright_high,win_y_high),(0,255,0), thickness=2) 
        
        ### TO-DO: Identify the number nonzero pixels in x and y within the window ###
        'Good_left/right_inds stores the indices of nonzero coordiate inside the windows'
        good_left_inds = ((nonzerox>win_xleft_low) & (nonzerox<win_xleft_high) &
                          (nonzeroy>win_y_low) & (nonzeroy<win_y_high)).nonzero()[0]
        good_right_inds = ((nonzerox>win_xright_low) & (nonzerox<win_xright_high) &
                           (nonzeroy>win_y_low) & (nonzeroy<win_y_high)).nonzero()[0]
        
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        ### TO-DO: If you found > minpix pixels, recenter next window ###
        ### (`right` or `leftx_current`) on their MEAN position ###
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
        
        

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    # Instead of list of lists, concatenate() would concvert it to 1D-aaray
    'The reason use **try** statement here is that let the code still move forward even when concatenate() fail.'
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        'How could we tell whether or not the above is implemented fully? If not, whats going to happen?'
        pass
    #After this code, left_lane_inds containt all the indices of pixel INSIDE the windows. 
    
    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    
    return leftx, lefty, rightx, righty, out_img


def fit_polynomial(binary_warped):
    # Find our lane pixels first
    leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped)

    ### TO-DO: Fit a second order polynomial to each using `np.polyfit` ###
    '''
    Take note of how we fit the lines above - while normally you CALCULATE A Y-VALUE FOR A
    GIVEN X,here we do the opposite. Why? Because we expect our lane lines to be (mostly) 
    vertically-oriented.
    '''
    left_fit = np.polyfit(lefty,leftx,2)
    left_fit_1d = np.poly1d(left_fit)
    right_fit = np.polyfit(righty,rightx,2)
    right_fit_1d = np.poly1d(right_fit)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    try:
        left_fitx=left_fit_1d(ploty)
        right_fitx=right_fit_1d(ploty)
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty

    ## Visualization ##
    # Colors in the left and right lane regions
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    # Plots the left and right polynomials on the lane lines
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    
    lane_pixel_ind=(leftx,lefty,rightx,righty)
    fit_1d=(left_fit_1d,right_fit_1d)
    polyfit_pts=(left_fitx,right_fitx,ploty)

    return out_img,lane_pixel_ind,fit_1d,polyfit_pts

def measure_curvature_real(binary_warped, xm_per_pix, ym_per_pix,lane_pixel_ind):
    leftx = lane_pixel_ind[0]
    lefty = lane_pixel_ind[1]
    rightx = lane_pixel_ind[2]
    righty = lane_pixel_ind[3]

    ### TO-DO: Fit a second order polynomial to each using `np.polyfit` ###
    '''
    Take note of how we fit the lines above - while normally you CALCULATE A Y-VALUE FOR A
    GIVEN X,here we do the opposite. Why? Because we expect our lane lines to be (mostly) 
    vertically-oriented.
    '''
    left_fit_cr = np.polyfit(ym_per_pix*lefty,xm_per_pix*leftx,2)
    left_fit_cr_1d = np.poly1d(left_fit_cr)
    right_fit_cr = np.polyfit(ym_per_pix*righty,xm_per_pix*rightx,2)
    right_fit_cr_1d = np.poly1d(right_fit_cr)
    

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    try:
        left_cr_fitx=left_fit_cr_1d(ploty)
        right_cr_fitx=right_fit_cr_1d(ploty)
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_cr_fitx = 1*ploty**2 + 1*ploty
        right_cr_fitx = 1*ploty**2 + 1*ploty

    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = ym_per_pix*np.max(ploty)
    
    ##### TO-DO: Implement the calculation of R_curve (radius of curvature) #####
    left_curverad = np.power(1+(2*left_fit_cr[0]*y_eval+left_fit_cr[1])**2,1.5)/np.absolute(2*left_fit_cr[0])
        ## Implement the calculation of the left line here
    right_curverad = np.power(1+(2*right_fit_cr[0]*y_eval+right_fit_cr[1])**2,1.5)/np.absolute(2*right_fit_cr[0])
        ## Implement the calculation of the right line here

    return left_curverad, right_curverad

def vehicle_offset(img,xm_per_pix,fit_1d):
    left_fit_1d = fit_1d[0]
    right_fit_1d = fit_1d[1]
    '''
    Calculate the offset of vehicle to center of the lane
    '''
    y_eval = np.max(img.shape[0])
    lane_ctr_x = np.mean(right_fit_1d(y_eval)-left_fit_1d(y_eval))
    vehicle_ctr_x = img.shape[1]/2
    offset_x = xm_per_pix*(vehicle_ctr_x-lane_ctr_x) # unit: meter
    
    return offset_x

def lane_highlight(undist, warped, Minv, polyfit_pts):
    left_fitx = polyfit_pts[0]
    right_fitx = polyfit_pts[1]
    ploty = polyfit_pts[2]
    
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (undist.shape[1], undist.shape[0])) 
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    
    return result

                        


# ## 1. Camera Calibration
# * Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
# Expected input & output
# <img src="./examples/undistort_output.png">
# 
# 

# ### 1.1. Camera distorsion calibration

# In[2]:


# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
nx = 9
ny = 6

objp = np.zeros((ny*nx,3), np.float32)
objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.

# Make a list of calibration images
images = glob.glob('camera_cal/calibration*.jpg')

# Step through the list and search for chessboard corners
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (nx,ny),None)

    # If found, add object points, image points
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)
    else: print('No chess board corner found and the filename is %s'%(fname))

    # Draw and display the corners
    img = cv2.drawChessboardCorners(img, (nx,ny), corners, ret)
    cv2.imshow('img',img)
    cv2.waitKey(500)

cv2.destroyAllWindows()


# ### 1.2. Distorsion image trail
# If the above cell ran sucessfully, you should now have `objpoints` and `imgpoints` needed for camera calibration.  Run the cell below to calibrate, calculate distortion coefficients, and test undistortion on an image!

# In[3]:


import pickle

# Test undistortion on an image
img = cv2.imread('camera_cal/test_image.jpg')
img_size = (img.shape[1], img.shape[0])
print("img_size is",img_size)

# Do camera calibration given object points and image points
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)


dst = cv2.undistort(img, mtx, dist, None, mtx)
cv2.imwrite('camera_cal/test_undist.jpg',dst)

# Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
# Create libary to store all the specs
dist_pickle = {}
dist_pickle["mtx"] = mtx
dist_pickle["dist"] = dist
# pickle.dump() is to dump/store/serialize the library dist_pickle = {} to the dest directory 
# so that we could retrieve this library later on OHTER python application.
pickle.dump( dist_pickle, open( "camera_cal/wide_dist_mtx.p", "wb" ) )
#dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
# Visualize undistortion
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
ax1.imshow(img)
ax1.set_title('Original Image', fontsize=30)
ax2.imshow(dst)
ax2.set_title('Undistorted Image', fontsize=30)
f.savefig('test_images_output/chessboard_undistort_output.jpg')


# ## 2. Pipeline Single Image
# 
# * Apply a distortion correction to raw images.
# * Use color transforms, gradients, etc., to create a thresholded binary image.
# * Apply a perspective transform to rectify binary image ("birds-eye view").
# * Detect lane pixels and fit to find the lane boundary.
# * Determine the curvature of the lane and vehicle position with respect to center.
# * Warp the detected lane boundaries back onto the original image.
# * Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

# ### 2.1. Provide an example of a distortion-corrected image.
# 
# <img src="./test_images/test1.jpg" width="500">

# ### 2.2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image. Provide an example of a binary image result.
# 
# <img src="./examples/binary_combo_example.jpg" width="500">

# #### Saturation Sacle & Gray Scale

# In[5]:


# Read in an image
#filename = 'test1.jpg'
filename = 'straight_lines1.jpg'
img = mpimg.imread('test_images/'+filename)

# Choose a Sobel kernel size
ksize = 3 # Choose a larger odd number to smooth gradient measurements

# Apply each of the thresholding functions
gradx = abs_sobel_thresh(img, orient='x', sobel_kernel=ksize, thresh=(20, 100))
grady = abs_sobel_thresh(img, orient='y', sobel_kernel=ksize, thresh=(20, 100))
mag_binary = mag_thresh(img, sobel_kernel=ksize, mag_thresh=(30, 100))

# Apply the color selection
rgb_threshold = 200
color_binary = color_select(img, rgb_threshold)

combined = np.zeros_like(mag_binary)
combined[((gradx == 1) & (grady == 1)) | (mag_binary == 1) | (color_binary == 1)] = 1

# Plot the result
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(img)
ax1.set_title('Original Image', fontsize=50)
ax2.imshow(combined, cmap='gray')
ax2.set_title('Combined.', fontsize=50)
f.savefig('test_images_output/binary_output.jpg')


# In[6]:


'''
# Plot the trail result for different criteria
f, a = plt.subplots(3, 2, figsize=(24, 24))
#f.tight_layout()
a[0][0].imshow(img)
a[0][0].set_title('Original Image', fontsize=50)
a[0][1].imshow(gradx, cmap='gray')
a[0][1].set_title('gradx', fontsize=50)
a[1][0].imshow(grady, cmap='gray')
a[1][0].set_title('grady', fontsize=50)
a[1][1].imshow(mag_binary, cmap='gray')
a[1][1].set_title('mag_binary', fontsize=50)
a[2][0].imshow(color_binary, cmap='gray')
a[2][0].set_title('color_binary', fontsize=50)
a[2][1].imshow(combined, cmap='gray')
a[2][1].set_title('Combined.', fontsize=50)
f.savefig('test_images_output/binary_output_clustered.jpg')
'''


# #### Region masking

# In[7]:


# Define the vertices of the region of masking

left_bottom = [200, 720]
left_upper = [550, 400]
right_upper = [780,400]
right_bottom = [1200, 720]
vertices = np.array([[left_bottom, right_bottom,right_upper, left_upper]],dtype=np.int32)
print('vertices is',vertices)

binary_mask = region_of_interest(combined, vertices)

# Plot the result
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(combined,cmap='gray')
ax1.set_title('Image Before Masking', fontsize=50)
ax2.imshow(binary_mask, cmap='gray')
ax2.set_title('Image after Masking', fontsize=50)
f.savefig('test_images_output/binary_output_masking.jpg')


# ### 2.3.Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.
# 
# | Source        | Destination   | 
# |:-------------:|:-------------:| 
# | 223, 719      | 320, 720      | 
# | 582, 456      | 320, 0        |
# | 694, 456      | 960, 0        |
# | 1087, 719     | 960, 720      |
# 
# 
# <img src="./examples/warped_straight_lines.jpg">

# In[8]:


# Read in the saved camera matrix and distortion coefficients
# These are the arrays you calculated using cv2.calibrateCamera()
dist_pickle = pickle.load( open( "camera_cal/wide_dist_mtx.p", "rb" ) )
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]

top_down, perspective_M, perspective_Minv, src = img_unwarp(binary_mask, mtx, dist)
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(img)
ax1.plot(src[::,0],src[::,1],'or',markersize=12,alpha=0.7)
ax1.set_title('Original Image', fontsize=50)
ax2.imshow(top_down,cmap='gray')
ax2.set_title('Undistorted and Warped Image', fontsize=50)
#plt.subplots_adjust(left=0.1, right=1, top=1, bottom=0.9)
f.savefig('test_images_output/perspectivetransform_output.jpg')


# ### 2.4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?
# 
# Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:
# 
# <img src="./examples/color_fit_lines.jpg" width="500">

# In[12]:


td_mark, lane_pixel_ind, fit_1d, polyfit_pts = fit_polynomial(top_down)


plt.imshow(td_mark)
plt.savefig('test_images_output/perspectivetransform_boxandline.jpg')


# ### 2.5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.
# 
# I did this in lines # through # in my code in my_other_file.py
# 
# [road marking](https://www.civil.iitb.ac.in/~vmtom/1100_LnTse/525_lnTse/plain/)
# 

# In[33]:


xm_per_pix = 3.7/640 # meters per pixel in x dimension
ym_per_pix = 12/720 # meters per pixel in y dimension
#ym_per_pix = 30/720 # meters per pixel in y dimension

# Calculate the radius of curvature in meters for both lane lines
left_curverad, right_curverad = measure_curvature_real(td_mark,xm_per_pix,ym_per_pix,lane_pixel_ind)

# Calculate the offset of vehicle to center of the lane
offset_x = vehicle_offset(img,xm_per_pix,fit_1d) #unit: meter


if offset_x > 0:
    output = "Vehicle is %.2fm right of center\n" %np.absolute(offset_x)
else:
    output = "Vehicle is %.2fm left of center\n" %np.absolute(offset_x)
output = output + 'Left lane curvature is %.2fm\nRight lane curvature is %.2fm '        %(left_curverad,right_curverad)
    
print(output)

f,ax1 = plt.subplots(1,1)
ax1.imshow(img)
ax1.set_title('Original Image', fontsize=30)
ax1.text(100,200, output, fontsize=15, color='white')
f.savefig('test_images_output/original_curvature_offset.jpg')


# ### 2.6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.
# 
# I implemented this step in lines # through # in my code in yet_another_file.py in the function map_lane(). Here is an example of my result on a test image:
# 
# <img src="./examples/example_output.jpg" width="500">

# In[35]:


plt.imshow(lane_highlight(img,top_down,perspective_Minv,polyfit_pts))
plt.savefig('test_images_output/original_highlightedlane.jpg')


# ## 3. Pipeline Multiple Images
# 
# Build the pipeline and run your solution on all test_images. Make copies into the `test_images_output` directory, and you can use the images in your writeup report.
# 
# Try tuning the various parameters, especially the low and high Canny thresholds as well as the Hough lines parameters.
# 
# TODO: Build your pipeline that will draw lane lines on the test_images
# 
# then save them to the test_images_output directory.

# In[3]:


import os
import shutil

# Create output directory
if os.path.exists("test_images_output/"):
    print('Output images will be saved to /test_images_output')
else: 
    print('Directory will be created: /test_images_output')
    os.makedirs("test_images_output/")

directory = os.listdir("test_images/")
    

for filename in directory:
    
    # Copy image to Output Folder
    shutil.copy('test_images/%s'%(filename),'test_images_output/%s'%(filename))
    
    # reading in an image
    image = cv2.imread('test_images/%s'%(filename))
    print('process %s ...'%(filename))

    # printing out some stats and plotting
    print('This image is:', type(image), 'with dimensions:', image.shape)
    plt.imshow(image) 
    
    ## Musk: Define the vertices of the region of masking
    color_mask = region_of_interest(image, vertices)
    
    #printing the greyscale image
    gray = grayscale(color_mask)
    
    # Define a kernel size for Gaussian smoothing / blurring.
    blur_gray = gaussian_blur(gray, kernel_size)
    
    # Run Canny edge detection
    edges = cv2.Canny(blur_gray, low_threshold, high_threshold)
    
    # Color Select
    rgb_threshold = [red_threshold, green_threshold, blue_threshold]
    color_select = np.copy(image)
    # Identify pixels below the threshold, and these that do NOT (below the threshold) will be blacked out.
    thresholds = (color_mask[:,:,0] < rgb_threshold[0])                 | (color_mask[:,:,1] < rgb_threshold[1])                 | (color_mask[:,:,2] < rgb_threshold[2]) 
    color_select[thresholds] = [0,0,0]
    
    # Hough Transform - Lane Finding
    line_image = hough_lines(edges, rho, theta, threshold, min_line_len, max_line_gap)

    # Remove the line from the Mask
    offset = -10
    vertices_offset = np.array([[[i-offset for i in left_bottom],                           [i-offset for i in right_bottom],                          [i-offset for i in right_upper],                          [i-offset for i in left_upper]]],dtype=np.int32)

    line_image = region_of_interest(line_image, vertices_offset)
    
    # Output the image
    image_out = weighted_img(line_image, image, α=0.8, β=1., γ=0.)
    plt.imshow(image_out)
    cv2.imwrite(os.path.join('test_images_output/%s_out.jpg'%(filename)),image_out)

    
    


# ## 4. Pipeline Video
# 
# You know what's cooler than drawing lanes over images? Drawing lanes over video!
# 
# We can test our solution on two provided videos:
# 
# `project_video.mp4`
# 
# **Note: if you get an import error when you run the next cell, try changing your kernel (select the Kernel menu above --> Change Kernel). Still have problems? Try relaunching Jupyter Notebook from the terminal prompt. Also, consult the forums for more troubleshooting tips.**
# 
# **If you get an error that looks like this:**
# ```
# NeedDownloadError: Need ffmpeg exe. 
# You can download it by calling: 
# imageio.plugins.ffmpeg.download()
# ```
# **Follow the instructions in the error message and check out [this forum post](https://discussions.udacity.com/t/project-error-of-test-on-videos/274082) for more troubleshooting tips across operating systems.**

# In[ ]:


# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML

def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image where lines are drawn on lanes)
    
    # printing out some stats and plotting
    # print('This image is:', type(image), 'with dimensions:', image.shape)
    # plt.imshow(image) 
    
    ## Musk: Define the vertices of the region of masking
    color_mask = region_of_interest(image, vertices)
    
    #printing the greyscale image
    gray = grayscale(color_mask)
    
    # Define a kernel size for Gaussian smoothing / blurring.
    blur_gray = gaussian_blur(gray, kernel_size)
    
    # Run Canny edge detection
    edges = cv2.Canny(blur_gray, low_threshold, high_threshold)
    
    # Color Select
    rgb_threshold = [red_threshold, green_threshold, blue_threshold]
    color_select = np.copy(image)
    # Identify pixels below the threshold, and these that do NOT (below the threshold) will be blacked out.
    thresholds = (color_mask[:,:,0] < rgb_threshold[0])                 | (color_mask[:,:,1] < rgb_threshold[1])                 | (color_mask[:,:,2] < rgb_threshold[2]) 
    color_select[thresholds] = [0,0,0]
    
    # Hough Transform - Lane Finding
    line_image = hough_lines(edges, rho, theta, threshold, min_line_len, max_line_gap)

    # Remove the line from the Mask
    offset = -10
    vertices_offset = np.array([[[i-offset for i in left_bottom],                           [i-offset for i in right_bottom],                          [i-offset for i in right_upper],                          [i-offset for i in left_upper]]],dtype=np.int32)

    line_image = region_of_interest(line_image, vertices_offset)
    
    # Output the image
    image_out = weighted_img(line_image, image, α=0.8, β=1., γ=0.)

    return image_out


# Let's try the one with the solid white lane on the right first ...

# In[ ]:


white_output = 'test_videos_output/project_video.mp4'
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
clip1 = VideoFileClip("test_videos/project_video.mp4").subclip(0,5)
## clip1 = VideoFileClip("test_videos/project_video.mp4")
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
get_ipython().run_line_magic('time', 'white_clip.write_videofile(white_output, audio=False)')


# Play the video inline, or if you prefer find the video in your filesystem (should be in the same directory) and play it in your video player of choice.

# In[ ]:


HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(white_output))


# In[ ]:




