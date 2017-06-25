# **Finding Lane Lines on the Road** 

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:

* Make a pipeline that finds lane lines on the road
* Apply the pipeline to find lane lines in videos


[//]: # (Image References)

[image1]: ./examples/original.png "original"
[image2]: ./examples/grayscale.png "grayscale"
[image3]: ./examples/gaussianblur.png "gaussianblur"
[image4]: ./examples/cannyedge.png "cannyedge"
[image5]: ./examples/maskededge.png "maskededge"
[image6]: ./examples/segmented.png "segmented hough"
[image7]: ./examples/finalimage.png "finalimage"

---

### Reflection

### 1. Pipeline

My pipeline consisted of the following steps:

1. Converted the images to grayscale;
2. Applied Gaussian smoothing
3. Found the canny edges
4. Created a masked edges in the region of interest of the canny edges
5. Ran Hough lines on the masked edges
6. Separated the lines into left and right lines
7. Averaged and extrapolated the seprated lines
8. Merged left and right lines
9. Drew the merged lines along with the original images

Steps 1 to 5 can be completed by the following helpers as provided:

```python
def grayscale(img):
	# Converted the images to grayscale
   return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def gaussian_blur(img, kernel_size):
	# Applied Gaussian smoothing
   return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    
def canny(img, low_threshold, high_threshold):
	# Found the canny edges
	return cv2.Canny(img, low_threshold, high_threshold)

def region_of_interest(img, vertices):
	#defining a blank mask to start with
	mask = np.zeros_like(img)   
    
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

def draw_lines(img, lines, color=[255, 0, 0], thickness=10):
	# draw lines on image
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)
            
def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
	# Ran Hough lines
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    return cv2.addWeighted(initial_img, α, img, β, λ)
```

In order to draw a single line on the left and right lanes (for steps 6 to 8), I added several functions before applying draw_lines() function:

```python
def separate_lines(lines):
    """ 
    separate lines by +/- slope.
    +: right lane
    -: left lane
    """
    right = []
    left = []
    for x1,y1,x2,y2 in lines[:, 0]:
        m = (float(y2) - y1) / (x2 - x1)
        if m >= 0: 
            right.append([x1,y1,x2,y2,m])
        else:
            left.append([x1,y1,x2,y2,m])
    
    return right, left

def reject_outliers(lines,  m1=0.4, m2=0.5):
    '''
    remove the lines for slope out of m1 to m2
    '''
    new_lines = []
    for x1,y1,x2,y2,m in lines:
        if (m >= m1 and m <= m2):
            new_lines.append([x1,y1,x2,y2])
    return new_lines

def merge_lines(lines):
    '''
    average and extrapolate lines
    '''
    linesa = np.asarray(lines)
    if linesa.size != 0:
        linesb = linesa.reshape(linesa.shape[0],1,linesa.shape[1])
        # average the lines coordinates
        merged_lines = np.mean(linesb,0)
    
        result_lines = merged_lines.reshape(merged_lines.shape[0],1,merged_lines.shape[1])
        result_lines = result_lines.astype('int')
    
        # return extrapolated lines
        return extraAndCombine(result_lines)
        
def extraAndCombine(lines):
    '''
    Extrapolate and combine lines into a single line
    '''
    # extrapolate the lines
    x=lines[0,0,0:3:2]
    y=lines[0,0,1:4:2]
    starts_ends_y = np.array([[[540,315]]])
    f = interpolate.interp1d(y, x, fill_value='extrapolate')
    starts_ends_x =np.array([[[f(starts_ends_y[0,0,0]), f(starts_ends_y[0,0,1])]]])
    
    # define the single line
    x1=starts_ends_x[0,0,0]
    y1=starts_ends_y[0,0,0]
    x2=starts_ends_x[0,0,1]
    y2=starts_ends_y[0,0,1]
    long_lines =np.array([[[x1,y1,x2,y2]]],dtype='int16')
    
    return long_lines

```

Now, let's take a look at how the pipeline works:

* Show the original image:
![alt text][image1]

* Grayscale image:
![alt text][image2]

* Gaussian blurred image:
![alt text][image3]

* Canny edges image:
![alt text][image4]

* Masked edges image:
![alt text][image5]

* Hough lines image:
![alt text][image6]

* Final image after **separate**,**average**, **extrapolate**, and **merge** function:
![alt text][image7]

* Video with lane lines detetion:

[![Final Video](https://img.youtube.com/vi/cJR8mmjCMuE/0.jpg)](https://www.youtube.com/watch?v=cJR8mmjCMuE)


### 2. Potential shortcomings with current pipeline


Current pipeline would work well when the image has good contrast to generate canny edges, thus one potential shortcoming would be that it can not detect the edges well when the constrast is poor, such as shadow of tree, or over the bridge.

Another shortcoming is current pipeline could not detect the curved line.


### 3. Suggest possible improvements to current pipeline

A possible improvement would be to find other ways to convert the image to get better constrast before applying canny edges, such as, converting RGB to HSV first (I have not tried this, and will try later)

Another potential improvement for curved lane lines could be to create polynominal lines instead of just creating the straight line.