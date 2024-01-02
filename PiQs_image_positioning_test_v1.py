#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 15:06:13 2023

@author: erri

The aim of this scriupt is to overlap camera photo over DEM
"""


# IMPORT PACKAGES
import os
import cv2
import numpy as np
from PIL import Image
import time
import math
import matplotlib.pyplot as plt
import PyPDF2
from PyPDF2 import PdfFileMerger, PdfFileReader, PdfFileWriter


home_dir = os.getcwd() # Home directory
report_dir = os.path.join(home_dir, 'output')
run_name = 'q20r1'
set_name = 'q20_2'
run_dir = os.path.join(home_dir, 'surveys')

# Convert matrices to images
# img1 = plt.imshow(channel, alpha=1.0)
# image = cv2.imread('/home/erri/Documents/PhD/Research/5_research_repos/PiQs_analysis/Photos/q07r1/Img0001.jpg', cv2.IMREAD_GRAYSCALE) # Image wet

# # q07r1
if run_name == 'q07r1':
    image = cv2.imread('/home/erri/Desktop/q07r1Img0001_ver2.jpg', cv2.IMREAD_GRAYSCALE) # image dry
    DEM = np.loadtxt('/home/erri/Documents/PhD/Research/5_research_repos/DoD_analysis/surveys/q07_1/matrix_bed_norm_q07_1s0.txt', skiprows=8)

if run_name == 'q10r1':
    image = cv2.imread('/home/erri/Desktop/q10r1Img0001_ver2.jpg', cv2.IMREAD_GRAYSCALE) # image dry
    DEM = np.loadtxt('/home/erri/Documents/PhD/Research/5_research_repos/DoD_analysis/surveys/q10_2/matrix_bed_norm_q10_2s0.txt', skiprows=8)

if run_name == 'q15r1':
    image = cv2.imread('/home/erri/Desktop/q15r1Img0001_ver2.jpg', cv2.IMREAD_GRAYSCALE) # image dry
    DEM = np.loadtxt('/home/erri/Documents/PhD/Research/5_research_repos/DoD_analysis/surveys/q15_2/matrix_bed_norm_q15_2s0.txt', skiprows=8)

if run_name == 'q20r1':
    image = cv2.imread('/home/erri/Desktop/q20r1Img0001_ver2.jpg', cv2.IMREAD_GRAYSCALE) # image dry
    DEM = np.loadtxt('/home/erri/Documents/PhD/Research/5_research_repos/DoD_analysis/surveys/q20_2/matrix_bed_norm_q20_2s0.txt', skiprows=8)


DEM = np.where(DEM==-999, np.nan, DEM)
DEM = np.repeat(DEM, 10, axis=1)

DEM    = DEM[:,DEM.shape[1]-1229:]



# # q07r1
if run_name == 'q07r1':
    # Define the transformation parameters
    scale = 0.198 # Enlargement scale
    dx = 3 # Shift in x direction
    dy = 33 # Shift in y direction
    rot_angle = -0.5

if run_name == 'q10r1':
    # Define the transformation parameters
    scale = 0.202 # Enlargement scale
    dx = 3 # Shift in x direction
    dy = 24 # Shift in y direction
    rot_angle = -0.4

if run_name == 'q15r1':
    # Define the transformation parameters
    scale = 0.200 # Enlargement scale
    dx = 0 # Shift in x direction
    dy = 30 # Shift in y direction
    rot_angle = -0.45

if run_name == 'q20r1':
    # Define the transformation parameters
    scale = 0.201 # Enlargement scale
    dx = 3 # Shift in x direction
    dy = 27 # Shift in y direction
    rot_angle = -0.45


def img_scaling_to_DEM(image, scale, dx, dy, rot_angle):

    # Create the transformation matrix
    M = np.float32([[scale, 0, dx], [0, scale, dy]])
    
    # Apply the transformation to img1 and store the result in img2
    rows, cols = image.shape
    image_rsh = cv2.warpAffine(image, M, (cols, rows))
    
    # Rotate the image
   
    M = cv2.getRotationMatrix2D((image_rsh.shape[1]/2, image_rsh.shape[0]/2), rot_angle, 1)
    image_rsh = cv2.warpAffine(image_rsh, M, (image_rsh.shape[1], image_rsh.shape[0]))
    
    # Trim zeros rows and columns due to shifting
    x_lim, y_lim = dx+int(cols*scale), dy+int(rows*scale)
    image_rsh = image_rsh[:y_lim, :x_lim]

    return image_rsh

image_rsh = img_scaling_to_DEM(image, scale, dx, dy, rot_angle)





# PLOT THE IMAGES
# img1 = plt.imshow(np.where(DEM ==0, np.nan, DEM), cmap='binary', origin='upper', alpha=1.0, vmin=-20, vmax=20, interpolation_stage='rgba')
img2 = plt.imshow(image_rsh, alpha=0.5, cmap='cool', origin='upper', vmin=-200, vmax=140, interpolation_stage='rgba')

# Set title and show the plot
plt.title(run_name)
plt.axis('off')
# plt.savefig(os.path.join(report_dir, 'surveys_images_overlappimg_test', run_name + '_DEM_photo_ovelapping.tiff'), dpi=600)
plt.savefig(os.path.join(report_dir, 'surveys_images_overlappimg_test', run_name + '_photo_ovelapping.tiff'), dpi=600)
# plt.savefig(os.path.join(report_dir, 'surveys_images_overlappimg_test', run_name + '_DEM_ovelapping.tiff'), dpi=600)


