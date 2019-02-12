import cv2
import matplotlib.pyplot as plt
import numpy as np
import math

#Function to display image
def display_image(img):
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img, cmap = 'gray')

#Apply 3x3 Smoothing filter
def smoothen(img):    
    f = np.full((3,3), 1)
    n = np.sum(f)
    height, width = img.shape
    smoothed = np.empty((height,width))
    for i in range(height):
        for j in range(width):
            #Set borders to the same values as the original image
            if i==0 or j==0 or i==(height-1) or j==(width-1):
                smoothed[i][j] = img[i][j]
            else:
                #Take slice of image the same size as the filter and convolve
                img_subarray = img[i-1:i+2, j-1:j+2]
                m = np.multiply(img_subarray, f)
                sum_m = np.sum(m)
                smoothed[i][j] = sum_m / n
                
    return smoothed

#Apply x-dervative
def x_derivative(img):    
    fx = np.array([-1,0,1])
    height, width = img.shape
    x = np.empty((height,width))
    for i in range(height):
        for j in range(width):
            if j == 0 or j == (width-1):
                x[i][j] = 0
            else:
                #Take horizontal slice of image the same size as the filter and convolve
                img_sub = img[i:i+1, j-1:j+2]
                x[i][j] = np.sum(np.multiply(img_sub, fx))
                
    return x

#Apply y-dervative
def y_derivative(img):    
    fy = np.array([[-1],[0],[1]])
    height, width = img.shape
    y = np.empty((height,width))
    for i in range(height):
        for j in range(width):
            if i == 0 or i == (height-1):
                y[i][j] = 0
            else:
                #Take vertical slice of image the same size as the filter and convolve
                img_sub = img[i-1:i+2, j:j+1]
                y[i][j] = np.sum(np.multiply(img_sub, fy))
                
    return y

#Apply 3x3 Gaussian filter
def gaussian(img):
    f = np.array([[1,2,1], [2,4,2], [1,2,1]])
    n = np.sum(f)
    height, width = img.shape
    detected = np.full((height,width), 1)
    for i in range(height):
        for j in range(width):
            #Set borders to the same values as the original image
            if i==0 or j==0 or i==(height-1) or j==(width-1):
                detected[i][j] = img[i][j]
            else:
                #Take slice of image the same size as the filter and convolve
                img_subarray = img[i-1:i+2, j-1:j+2]
                m = np.multiply(img_subarray, f)
                sum_m = np.sum(m)
                detected[i][j] = sum_m;
                
    return detected

def non_max_suppress(img):
    height, width = img.shape
    s = np.full((height,width), 0)
    for i in range(height):
        for j in range(width):
            if i == 0 or i == height-1 or j == 0 or j == width-1:
                continue
            else:
                if (img[i-1][j-1] > img[i][j]) or (img[i-1][j] > img[i][j]) or (img[i-1][j+1] > img[i][j]) or (img[i][j-1] > img[i][j]) or (img[i][j+1] > img[i][j]) or (img[i+1][j-1] > img[i][j]) or (img[i+1][j] > img[i][j]) or (img[i+1][j+1] > img[i][j]):
                    s[i][j] = 0
                else:
                    s[i][j] = img[i][j]
                    
    return s

#Input image
filename = input('Enter image path: ')
img = cv2.imread(filename, 0)

height, width = img.shape

display_image(img)

#Smoothen image
smoothed = smoothen(img)
display_image(np.round(smoothed))

#x-derivative of image
dx = x_derivative(smoothed)

#Display x-derivative in the range [0,255]
display_image(dx + 128)

#y-derivative of image
dy = y_derivative(smoothed)

#Display y-derivative in the range [0,255]
display_image(dy + 128)

#Edge map or Gradient magnitude
gradient_magnitude = np.sqrt(np.square(dx) + np.square(dy))
display_image(gradient_magnitude)
cv2.imwrite('downloads/edges.png', gradient_magnitude)

edges = cv2.Canny(img, 0, 255)

#display_image(edges)
cv2.imwrite('downloads/canny.png', edges)

#Hough transform
hough_space = np.full((1200,180), 0)
count = np.full((1200,180), 0)
for i in range(height):
    for j in range(width):
        if edges[i][j] != 0:
            for theta in range(-90,90):
                rho = (i * math.cos(math.radians(theta))) + (j * math.sin(math.radians(theta)))
                hough_space[int(round(rho)+500)][int(round(theta)+90)] = 255
                count[int(round(rho+500))][int(round(theta))+90] += 1
                
#display_image(hough_space)
display_image(count)
cv2.imwrite('downloads/count.png', count)

h, w = count.shape

#Peak detection using threshold
new_peaks = np.full((h,w), 0)
for i in range(h):
    for j in range(w):
        if count[i][j] > 0.80*count.max():
            new_peaks[i][j] = count[i][j]
        else:
            new_peaks[i][j] = 0

#Number of peaks detected
np.count_nonzero(new_peaks)

#Re-map peak values to image space
detected_lines = np.full((height, width), 0)
for rho in range(1200):
    for theta in range(-90,90):
        if new_peaks[rho][theta+90] != 0:
            for x in range(height):
                for y in range(width):
                    if round(((x * math.cos(math.radians(theta))) + (y * math.sin(math.radians(theta))))) + 500 == rho:
                        detected_lines[x][y] = 255
                        
display_image(detected_lines)
cv2.imwrite('downloads/lines.png', detected_lines)
