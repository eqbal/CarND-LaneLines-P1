#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math
import os


class LaneFinder(object):

  def __init__(self, image):
    self.original = image
    self.image    = image
    self.width    = image.shape[1]
    self.height   = image.shape[0]

  CANNY_LOW_THRESHOLD = 80
  CANNY_HIGH_THRESHOLd = 200
  RHO = 2
  THETA = np.pi / 180
  THRESHOLD = 20
  MIN_LINE_LENGTH = 20
  MAX_LINE_GAP = 10
  GAUSSIAN_KERNEL = 5

  def grayscale(self):
    self.image = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)

  def canny(self, low_threshold, high_threshold):
    """Applies the Canny transform"""
    self.image = cv2.Canny(self.image, low_threshold, high_threshold)

  def gaussian_blur(self, kernel_size):
    """Applies a Gaussian Noise kernel"""
    self.image = cv2.GaussianBlur(self.image, (kernel_size, kernel_size), 0)

  def region_of_interest(self):
    #defining a blank mask to start with
    mask = np.zeros_like(self.image)

    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    ignore_mask_color = self.get_ignore_mask_color()

    #define vertices for the needed region (left and right lane)
    vertices = self.generate_vertices()

    #filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    #returning the image only where mask pixels are nonzero
    self.image = cv2.bitwise_and(self.image, mask)

  def get_ignore_mask_color(self):
    if len(self.image.shape) > 2:
      channel_count = self.image.shape[2]  # i.e. 3 or 4 depending on your image
      ignore_mask_color = (255,) * channel_count
    else:
      ignore_mask_color = 255
    return ignore_mask_color

  def generate_vertices(self):
    vertices = np.array([[
      (0,self.height),
      ((self.width/2), ((self.height/2)+10)),
      ((self.width/2), ((self.height/2)+10)),
      (self.width,self.height)
    ]], dtype=np.int32)
    return vertices

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

    initial_img * a + img * b + g
    NOTE: initial_img and img must be the same shape!
    """
    self.image = cv2.addWeighted(initial_img, a, self.image, b, g)

#reading in an image
image = mpimg.imread('CarND-LaneLines-P1/test_images/solidWhiteRight.jpg')
#printing out some stats and plotting
print('This image is:', type(image), 'with dimesions:', image.shape)

finder = LaneFinder(image)
finder.grayscale()
finder.region_of_interest()

plt.imshow(finder.image)  #call as  to show a grayscaled image
# plt.imshow(image_after_grayscale, cmap='gray')
plt.show()



