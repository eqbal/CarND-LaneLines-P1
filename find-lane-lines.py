#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math
import os


class LaneFinder(object):

  CANNY_LOW_THRESHOLD = 64
  CANNY_HIGH_THRESHOLd = 192
  RHO = 1
  THETA = np.pi / 180
  THRESHOLD = 32
  MIN_LINE_LENGTH = 1
  MAX_LINE_GAP = 200
  GAUSSIAN_KERNEL = 5

  def __init__(self, image):
    self.original = image
    self.image    = image
    self.width    = image.shape[1]
    self.height   = image.shape[0]

  def call(self):
    self.grayscale()
    self.gaussian_blur()
    self.canny()
    self.remove_noise()
    self.region_of_interest()
    self.hough_lines()
    self.weighted_img()

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
      ((self.width*3/8), (self.height*5/8)),
      ((self.width*5/8), (self.height*5/8)),
      (self.width,self.height)
    ]], dtype=np.int32)

    return vertices

  def draw_lines(self, img, lines, color=[255, 0, 0], thickness=2):
    for line in lines:
      for x1,y1,x2,y2 in line:
        cv2.line(img, (x1, y1), (x2, y2), color, thickness)

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

    line_img = np.zeros((self.height, self.width, 3), dtype=np.uint8)

    if self.lines != None:
      lines2 = []
      for line in self.lines:
        for x1,y1,x2,y2 in line:
          k = float(y2-y1)/(x2-x1)
          if y1 > y2:
            x1 = int(x2 + (self.height-y2)/k)
            y1 = self.height
          elif y1 < y2:
            x2 = int(x1 + (self.height-y1)/k)
            y2 = self.height
          lines2.append([[x1, y1, x2, y2]])

      self.draw_lines(line_img, lines2, thickness=10)

    self.image = line_img


  def weighted_img(self, a=0.8, b=1., g=0.):
    self.image = cv2.addWeighted(self.original, a, self.image, b, g)

#reading in an image
image = mpimg.imread('CarND-LaneLines-P1/test_images/solidWhiteRight.jpg')
#printing out some stats and plotting
print('This image is:', type(image), 'with dimesions:', image.shape)

finder = LaneFinder(image)
finder.call()

plt.imshow(finder.image)

plt.show()



