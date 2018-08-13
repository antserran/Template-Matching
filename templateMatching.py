# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 12:26:57 2018

@author: Antonio Serrano

"""
import cv2 # Importing OpenCV library
from matplotlib import pyplot as plt # Importing matplotlib for displaying


def show_results (img,template,matching,detection):
    
    """ It shows the result images in a 2x2 grid
    """
    
    # Displaying original image
    plt.subplot(2,2,1),plt.imshow(img,cmap = 'gray')
    plt.title('Original image'), plt.xticks([]), plt.yticks([])
    
    # DIsplaying template image
    plt.subplot(2,2,2),plt.imshow(template,cmap = 'gray')
    plt.title('Template image'), plt.xticks([]), plt.yticks([])
    
    # Displaying matching image 
    plt.subplot(2,2,3),plt.imshow(matching,cmap = 'gray')
    plt.title('Matching result'), plt.xticks([]), plt.yticks([])
    
    # Displaying original image with detection
    plt.subplot(2,2,4),plt.imshow(detection,cmap = 'gray')
    plt.title('Eye detection'), plt.xticks([]), plt.yticks([])

def template_matching(img, template, method):

    """ It perform template matching technique given a image, template and
    method
    
    Method can be: 
        method = cv2.TM_SQDIFF
        method = cv2.TM_SQDIFF_NORMED
        method = cv2.TM_CCOEFF
        method = cv2.TM_CCOEFF_NORMED 
        method = cv2.TM_CCORR
        method = cv2.TM_CCORR_NORMED
    """
    
    # Calculating matching image, consisting of moving the template around the
    # image and comparing template and original images
    matching = cv2.matchTemplate(image=img, templ=template, method=method)
    
    # Getting the maximum/minimum values and its positions of the mathing image,
    # that is, where the original image matches the best with the template image
    # 'max_point' is the point where the object we are looking for should be
    # Note that if the method used is cv2.TM_SQDIFF, then 'min_point' is the point 
    # where the object shoud be
    min_value, max_value, min_point, max_point = cv2.minMaxLoc(matching)

    aux = img.copy() # Aux image necessary to draw rectangle over original image
    
    # Getting dimensions of template in order to calculate rectangle's size
    h,w = template.shape 
    
    # Calculating rectangle dimensions
    top_left = max_point
    bottom_right = (top_left[0] + w, top_left[1] + h)
    
    # Drawing rectangle around eye zone detected
    detection = cv2.rectangle(img=aux, pt1=top_left, pt2=bottom_right, color=255, thickness=2)
    
    return matching, detection


if __name__ == "__main__" :

    # Reading original image in grayscale (0)
    image = cv2.imread('image.jpg', 0)
    
    # Reading template image in grayscale (0)
    template = cv2.imread('eye.jpg',0)
    
    # Performing template matching
    matching, detection = template_matching(image, template, method=cv2.TM_CCOEFF)
    
    # Showing results
    show_results(image, template, matching, detection)