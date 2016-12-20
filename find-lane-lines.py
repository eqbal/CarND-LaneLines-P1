#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math
import os


class LaneFinder(object):
  def __init__(self, image):
    self.image = image

  def grayscale(self):
    return cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)

  def canny(self, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(self.image, low_threshold, high_threshold)

  def gaussian_blur(self, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(self.image, (kernel_size, kernel_size), 0)

  def region_of_interest(self, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(self.image)

    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(self.image.shape) > 2:
      channel_count = self.image.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
      ignore_mask_color = 255

    #filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(self.image, mask)
    return masked_image


  def draw_lines(self, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    for line in lines:
      for x1,y1,x2,y2 in line:
        cv2.line(self.image, (x1, y1), (x2, y2), color, thickness)

  def hough_lines(self, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(self.image, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((self.image.shape[0], self.image.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

  # Python 3 has support for cool math symbols.

  def weighted_img(self, initial_img, a=0.8, b=1., g=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, a, self.image, b, g)

#reading in an image
image = mpimg.imread('test_images/solidWhiteRight.jpg')
#printing out some stats and plotting
print('This image is:', type(image), 'with dimesions:', image.shape)

finder = LaneFinder(image)
image_after_grayscale = finder.grayscale()

# plt.imshow(image)  #call as  to show a grayscaled image
plt.imshow(image_after_grayscale, cmap='gray')
plt.show()



