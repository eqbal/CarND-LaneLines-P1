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

  def canny(self):
    """Applies the Canny transform"""
    self.image = cv2.Canny(self.image, self.CANNY_LOW_THRESHOLD, self.CANNY_HIGH_THRESHOLd)

  def remove_noise(self):
    self.image = cv2.dilate(self.image, cv2.getStructuringElement(cv2.MORPH_DILATE, (5, 5)))

  def gaussian_blur(self):
    """Applies a Gaussian Noise kernel"""
    self.image = cv2.GaussianBlur(self.image, (self.GAUSSIAN_KERNEL, self.GAUSSIAN_KERNEL), 0)

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
      channel_count = self.image.shape[2]
      ignore_mask_color = (255,) * channel_count
    else:
      ignore_mask_color = 255
    return ignore_mask_color

  def generate_vertices(self):
    vertices = np.array([[
      (0,self.height),
      ((self.width/2), ((self.height/2)+30)),
      ((self.width/2), ((self.height/2)+30)),
      (self.width,self.height)
    ]], dtype=np.int32)
    return vertices

  def draw_lines(self, lines, color=[255, 0, 0], thickness=2):
    for line in lines:
      for x1,y1,x2,y2 in line:
        cv2.line(self.image, (x1, y1), (x2, y2), color, thickness)

  def hough_lines(self):
    self.lines = cv2.HoughLinesP(
        self.image,
        self.RHO,
        self.THETA,
        self.THRESHOLD,
        np.array([]),
        minLineLength=self.MIN_LINE_LENGTH,
        maxLineGap=self.MAX_LINE_GAP
    )

  def draw_hough_lines(self):
    line_img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
    self.draw_lines(line_img, self.lines)
    self.image = line_img

  def weighted_img(self, initial_img, a=0.8, b=1., g=0.):
    self.image = cv2.addWeighted(initial_img, a, self.image, b, g)

#reading in an image
image = mpimg.imread('CarND-LaneLines-P1/test_images/solidWhiteRight.jpg')
#printing out some stats and plotting
print('This image is:', type(image), 'with dimesions:', image.shape)

finder = LaneFinder(image)

finder.grayscale()
plt.subplot(231)
plt.imshow(finder.image, cmap='gray')

finder.gaussian_blur()
plt.subplot(232)
plt.imshow(finder.image, cmap='gray')

finder.canny()
plt.subplot(233)
plt.imshow(finder.image, cmap='gray')

finder.remove_noise()
plt.subplot(234)
plt.imshow(finder.image, cmap='gray')

finder.region_of_interest()
plt.subplot(235)
plt.imshow(finder.image, cmap='gray')

finder.hough_lines()
print finder.lines
plt.show()



