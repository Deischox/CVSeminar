import matplotlib.pyplot as plt

import matplotlib.lines as mlines
import numpy as np
import math
import random
from skimage import filters, color
from PIL import Image
import cv2

lines = []
fig, axs = plt.subplots(1, 2)

def calculateLines(a, b):
    x = np.linspace(-5,5,100)
    y = -a*x+b
    lines.append(y)
    axs[1].plot(x,y,'-r')

def intersectPoint(a,b,a1,b1):
    x = (b-b1)/(a-a1)
    y = -a*x+b

    w = np.linspace(-5, 5, 100)

    lines = x*w+y

    axs[0].plot(w,lines,'-b')

def houghlines(x,y, edge_image):
    edge_height, edge_width = edge_image.shape[:2]
    d = np.sqrt(np.square(edge_height) + np.square(edge_width))

    thetas = np.arange(-90,90)
    rhos = np.arange(-d,d)

    accumulator = np.zeros((len(rhos), len(rhos)))

    for y in range(edge_height):
        for x in range(edge_width):
            if edge_image[x][y] != 0:
                for thet in range(len(thetas)):
                    rho = (edge_image[x][y] * math.cos(thetas[thet])) + (edge_image[x][y] * math.sin(thetas[thet]))
                    rho_indx = np.argmin(np.abs(rhos-rho))
                    axs[1].plot(thetas[thet],rho_indx,'ro')
                    accumulator[rho_indx][thet] += 1
def line_detection_non_vectorized(image, edge_image, num_rhos=180, num_thetas=180, t_count=220):
  edge_height, edge_width = edge_image.shape[:2]
  edge_height_half, edge_width_half = edge_height / 2, edge_width / 2
  #
  d = np.sqrt(np.square(edge_height) + np.square(edge_width))
  dtheta = 180 / num_thetas
  drho = (2 * d) / num_rhos
  #
  thetas = np.arange(0, 180, step=dtheta)
  rhos = np.arange(-d, d, step=drho)
  #
  cos_thetas = np.cos(np.deg2rad(thetas))
  sin_thetas = np.sin(np.deg2rad(thetas))
  #
  accumulator = np.zeros((len(rhos), len(rhos)))
  #
  figure = plt.figure(figsize=(12, 12))
  subplot1 = figure.add_subplot(1, 4, 1)
  subplot1.imshow(image, cmap="gray")
  subplot2 = figure.add_subplot(1, 4, 2)
  subplot2.imshow(edge_image, cmap="gray")
  subplot3 = figure.add_subplot(1, 4, 3)
  subplot3.set_facecolor((0, 0, 0))
  subplot4 = figure.add_subplot(1, 4, 4)
  subplot4.imshow(image, cmap="gray")
  #
  for y in range(edge_height):
    for x in range(edge_width):
      if edge_image[y][x] != 0:
        edge_point = [y - edge_height_half, x - edge_width_half]
        ys, xs = [], []
        for theta_idx in range(len(thetas)):
          rho = (edge_point[1] * cos_thetas[theta_idx]) + (edge_point[0] * sin_thetas[theta_idx])
          theta = thetas[theta_idx]
          rho_idx = np.argmin(np.abs(rhos - rho))
          accumulator[rho_idx][theta_idx] += 1
          ys.append(rho)
          xs.append(theta)
        subplot3.plot(xs, ys, color="white", alpha=0.1)

  for y in range(accumulator.shape[0]):
    for x in range(accumulator.shape[1]):
      if accumulator[y][x] > t_count:
        rho = rhos[y]
        theta = thetas[x]
        a = np.cos(np.deg2rad(theta))
        b = np.sin(np.deg2rad(theta))
        x0 = (a * rho) + edge_width_half
        y0 = (b * rho) + edge_height_half
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        subplot3.plot([theta], [rho], marker='o', color="yellow")
        subplot4.add_line(mlines.Line2D([x1, x2], [y1, y2]))

  subplot3.invert_yaxis()
  subplot3.invert_xaxis()

  subplot1.title.set_text("Original Image")
  subplot2.title.set_text("Edge Image")
  subplot3.title.set_text("Hough Space")
  subplot4.title.set_text("Detected Lines")
  plt.show()


def start():
    axs[0].plot(1,4,'ro')
    axs[0].plot(3,4,'ro')
    axs[0].plot(2,4,'ro')

    axs[0].set_xlim([0,5])
    axs[0].set_ylim([0,5])

    calculateLines(1,4)
    calculateLines(3,4)
    calculateLines(2,4)

    intersectPoint(1,4,3,4)


    plt.show()

img2 = plt.imread("small.png")
img2 = color.rgb2gray(img2)
edge = filters.sobel(img2)
#edge = img2

for x in range(0,len(edge)):
    for y in range(0,len(edge[0])):
        if edge[x][y] > 0.2:
            edge[x][y] = 1
        else:
            edge[x][y] = 0

plt.imshow(edge)

line_detection_non_vectorized(img2, edge, t_count=500)
plt.show()
