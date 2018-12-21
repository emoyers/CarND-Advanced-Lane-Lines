## Advanced Lane Finding Project Emilio Moyers

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

[image0]: ./camera_cal/calibration1.jpg "original chessboard"
[image1]: ./writeup_images/undistorted.jpg "Undistorted chessboard"
[image2]: ./writeup_images/undistorted_car.jpg "Road Transformed"
[image3]: ./writeup_images/binary.jpg "Binary Example"
[image4]: ./writeup_images/binary_w_lines.jpg "Warp Example"
[image5]: ./writeup_images/perspective.jpg "Warp Example"
[image6]: ./writeup_images/perspective_squares.jpg "Perspective with squares"
[image7]: ./writeup_images/perspective_region.jpg "Perspective with region"
[image8]: ./writeup_images/final_image.jpg "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. [Here](https://github.com/emoyers/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a the writeup for the Advanced Lane Finding Project.  

You're reading it hahaha!

### Camera Calibration

The code for this step is contained in the **4th** code cell of the IPython notebook located in `P2.ipynb`.  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `obj_points` is just a replicated array of coordinates, and `object_points` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `image_points` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `object_points` and `image_points` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result:

![alt text][image0]
**Original image**

![alt text][image1]
**Undistorted image**

---

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of saturation (S) in the HSL color space, gradient thresholds in x direction and illumination (I) also from HSL to generate a binary image:
```python
combined_binary[((saturation_binary == 1) | (sobel_binary == 1)) & (light_binary == 1)] = 1
```
This part of the code is in the **7th** code cell of the IPython notebook located in `P2.ipynb`.  Here's an example of my output for this step.  (note: this is not actually from one of the test images)

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `prespective_transform()`, which appears in the **9th** code cell of the IPython notebook located in `P2.ipynb`.  The `prespective_transform()` function takes as inputs an image (`img`), as well as source (`source_point`) and destination (`destination_point`) points.  I chose the hardcode the source and destination points in the following manner:

```python
#Source points
x1_s = 200
x2_s = 555
x3_s = 732
x4_s = 1100
y_top_s = 720
y_low_s = 480
#generating the array for source points
src = np.float32([[x1_s, y_top_s],[x2_s, y_low_s],[x3_s, y_low_s],[x4_s, y_top_s]])

#destination points
x_low_d = 290
x_top_d = 990
y_top_d = 720
y_low_d = 0
#generating the array for destination points
dst = np.float32([[x_low_d, y_top_d],[x_low_d, y_low_d],[x_top_d, y_low_d],[x_top_d, y_top_d]])
```

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]
**Original Binary image with lines in red**

![alt text][image5]
**Perspective Transformed image**

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I used
```python
 histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
```
To calculated the 2 greater regions of x to find where the lines are located in the image.
After that I used sliding rectangles technique to follow the line within an image, this part of the code is located in the **11th** code cell of the IPython notebook located in `P2.ipynb`. This technique consist of two functions:
* `fit_polynomial()` which input is the binary image (`binary_warped`) and its outputs are the points of x in which the line is through the whole `left_fitx, right_fitx` y axis and the coefficients of the curve that pass through all these point for both lines `left_fit, right_fit`.
* `find_lane_pixels()` this function is called by `fit_polynomial()` and return similar output `left_fitx, right_fitx` and `left_fit, right_fit`. Also is in charge of drawing the rectangles.

The result I get is something like this:
![alt text][image6]

Due to the process of `fit_polynomial()` function is not very efficient for a real time scenario another approach is used after getting the first values of  `left_fitx, right_fitx` and `left_fit, right_fit`. Using this information I calculate a region where most like the line would be for this I use two functions as well, this part of the code is found in the **13th** code cell of the IPython notebook located in `P2.ipynb`:
* `fit_poly` find new points in X within the whole Y axis for the 2 lines (`left_fitx, right_fitx`), calculate the curvature of the line (`left_curve_rad_m, right_curve_rad_m`) and also new coefficients (`left_fit, right_fit`).
* `search_around_poly` calls `fit_poly` to get `left_fitx, right_fitx, left_fit, right_fit, left_curve_rad_m, right_curve_rad_m` for drawing the region where most likely the line will be located.

The result I get is something like this:
![alt text][image7]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

As mention in the previous point (5) the **calculation of the radius of curvature** is done in the **13th** code cell of the IPython notebook located in `P2.ipynb` within the function `fit_poly`.

```python
##### Implement the calculation of R_curve (radius of curvature) from f(y) = Ay^2 + By + C  f'(y) = 2Ay + B  f''(y) = 2A#####
#Rcurve = ([1+ (dx/dy)^2]^(3/2))/(|(d2x/dy2)|) = ([1+ (2Ay + B)^2]^(3/2))/(|2A|)
left_curve_rad = np.power((1+np.power(2*left_fit[0]*y_eval+left_fit[1],2)),(3/2))/np.absolute(2*left_fit[0])
right_curve_rad = np.power((1+np.power(2*right_fit[0]*y_eval+right_fit[1],2)),(3/2))/np.absolute(2*right_fit[0])   
#print(left_curve_rad, right_curve_rad)

### Real world calculation
left_curve_rad_m = np.power((1+np.power(2*left_fit_m[0]*y_eval_m*+left_fit_m[1],2)),(3/2))/np.absolute(2*left_fit_m[0])  
right_curve_rad_m = np.power((1+np.power(2*right_fit_m[0]*y_eval_m+right_fit_m[1],2)),(3/2))/np.absolute(2*right_fit_m[0])   
#print(left_curve_rad_m, 'm', right_curve_rad_m, 'm')
```

For the **position of the vehicle with respect to center** is located in the **17th** code cell of the IPython notebook located in `P2.ipynb` with the function `main_call`. The code below shows the calculation:

```python
#Calculate the offset to the center line
xm_per_pix = 3.7/700 # meters per pixel in x dimension
center_between_lines = ((right_fitx[len(right_fitx)-1]-left_fitx[len(left_fitx)-1])/2)+left_fitx[len(left_fitx)-1]
Offset_center_lines = final_binary_image.shape[1]/2 - center_between_lines
left_line.line_base_pos = right_line.line_base_pos = Offset_center_lines*xm_per_pix
```

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in the **15th** code cell of the IPython notebook located in `P2.ipynb` in the function `draw_lines_region()`.  Here is an example of my result on a test image:
(Note: in the image you can visualize the **radius of curvature** of each line and the **position of the vehicle with respect to center** )

![alt text][image8]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./output_images/project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

When I integrated the pipeline to process all frames of the video I faced the problem that in some point the green region got lost that is why I needed to implement and algorithm to check if the current are good or not. I made this in the **17th** code cell of the IPython notebook located in `P2.ipynb` with the functions:
* `average_coeficients`, it takes the average of the n store coefficients of each line, right and left.
* `line_distance_comparison`, compare if the line calculated are parallel of not, by comparing the distance between three points of each line in x, at x=0, x=maximum value of x over 2 and x=maximum value.
* `main_call`, uses  `line_distance_comparison`and `average_coeficients` to figure it out if the calculated coeficients are useful or not. If there aren't, it calls `fit_polynomial()` to recalculte the histogram and new accurate coefficients and point on x for each line.
An abstract of the code is here:
```python
#Compare the distance in three point bottom, middle and top in Y axis and decide if the they meet the treshholds
if  left_line.loops_counter == 0 & right_line.loops_counter == 0 :
    left_line.n_fits = np.array([left_line.current_fit])
    right_line.n_fits = np.array([right_line.current_fit])
    left_line.loops_counter += 1
    right_line.loops_counter += 1
elif ((left_line.loops_counter < n_loops) & (right_line.loops_counter < n_loops)):
    if line_distance_comparison(left_fitx, right_fitx):
        #add more elements to the matrix of n line coeficcieents
        left_line.n_fits = np.insert(left_line.n_fits, 0,left_line.current_fit, axis=0)
        right_line.n_fits = np.insert(right_line.n_fits, 0,right_line.current_fit, axis=0)
        #increment by 1 until n loops
        left_line.loops_counter += 1
        right_line.loops_counter += 1
        left_line.error_fit = 0
        right_line.error_fit = 0
    else:
        #Calculate the value of the x in 0, mid point x and greater point x with the average coeficients for both lines
        y_3_point = np.array([0, final_binary_image.shape[0]//2, final_binary_image.shape[0]-1])

        ### Calc both polynomials using ploty, left_fit and right_fit ###
        left_fitx_3_points = left_line.best_fit.item(0)*y_3_point**2 + left_line.best_fit.item(1)*y_3_point + left_line.best_fit.item(2)
        right_fitx_3_points = right_line.best_fit.item(0)*y_3_point**2 + right_line.best_fit.item(1)*y_3_point + right_line.best_fit.item(2)

        if ((left_fitx[len(left_fitx)-1]) > (left_fitx_3_points.item(2) - between_same_lines_threshhold)) &\
        ((left_fitx[len(left_fitx)-1]) < (left_fitx_3_points.item(2) + between_same_lines_threshhold)) &\
        ((left_fitx[len(left_fitx)//2]) > (left_fitx_3_points.item(1) - between_same_lines_threshhold)) &\
        ((left_fitx[len(left_fitx)//2]) < (left_fitx_3_points.item(1) + between_same_lines_threshhold)) &\
        ((left_fitx[0]) > (left_fitx_3_points.item(0) - between_same_lines_threshhold)) &\
        ((left_fitx[0]) < (left_fitx_3_points.item(0) + between_same_lines_threshhold)):
            #add more elements to the matrix of n line coeficcieents
            left_line.n_fits = np.insert(left_line.n_fits, 0,left_line.current_fit, axis=0)
            left_line.loops_counter += 1
            left_line.error_fit = 0      
#calculate the average of the coeficients
average_coeficients(left_line.n_fits, right_line.n_fits)
```
