#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 30 14:11:51 2022

@author: erri

The aims of this script are:
    1. Produce overlapping maps of the mein measures taken in the lab.
       In particular this script overlaps DoD, BAA and DEM map
    2. Perform analysis on the overlapped data to investigate the relations
       between morphological changes, bed elevation end bedload activity

INPUT (as .npy binary files):
    DoD_stack1 : 3D numpy array stack
        Stack on which DoDs are stored as they are, with np.nan
    DoD_stack1_bool : 3D numpy array stack
        Stack on which DoDs are stored as -1, 0, +1 data, also with np.nan
OUTPUTS:
    For each run the scrip gives you a map where the envelope of the Bedload
    Active Area and the corresponding DoD are overlapped
    
"""
# IMPORT LIBRARIES
import os
import cv2
import numpy as np
from PIL import Image
import time
import math
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import scipy.ndimage
import PyPDF2
from PyPDF2 import PdfFileMerger, PdfFileReader, PdfFileWriter

from scipy import ndimage
def downsample_matrix_interpolation(matrix, factor):
    # Get the shape of the original matrix
    height, width = matrix.shape

    # Calculate the new dimensions after downsampling
    new_height = height // factor
    new_width = width // factor

    # Create a grid of coordinates for the new downsampling points
    row_indices = np.linspace(0, height - 1, new_height)
    col_indices = np.linspace(0, width - 1, new_width)

    # Perform bilinear interpolation to estimate the values at new points
    downsampled_matrix = ndimage.map_coordinates(matrix, 
                                                 np.meshgrid(row_indices, col_indices),
                                                 order=1,
                                                 mode='nearest')

    # Reshape the downsampled matrix to the new dimensions
    downsampled_matrix = downsampled_matrix.reshape(new_height, new_width)

    return downsampled_matrix
    
# FOLDER SETUP
home_dir = os.getcwd() # Home directory
report_dir = os.path.join(home_dir, 'output')
run_dir = os.path.join(home_dir, 'surveys')
DoDs_folder = os.path.join(home_dir, 'DoDs', 'DoDs_stack') # Input folder

#%%

run_names = ['q07r1','q07r2','q07r3','q07r4','q07r5','q07r6','q07r7','q07r8','q07r9'
        ,'q10r1','q10r2','q10r3','q10r4','q10r5','q10r6','q10r7','q10r8','q10r9'
        ,'q15r1','q15r2','q15r3','q15r4','q15r5','q15r6','q15r7','q15r8','q15r9'
        ,'q20r1','q20r2','q20r3','q20r4','q20r5','q20r6','q20r7','q20r8','q20r9']
set_names = ['q07_1','q07_1','q07_1','q07_1','q07_1','q07_1','q07_1','q07_1','q07_1'
        ,'q10_2','q10_2','q10_2','q10_2','q10_2','q10_2','q10_2','q10_2','q10_2'
        ,'q15_2','q15_2','q15_2','q15_2','q15_2','q15_2','q15_2','q15_2','q15_2'
        ,'q20_2','q20_2','q20_2','q20_2','q20_2','q20_2','q20_2','q20_2','q20_2']


# run_names = ['q07r1']
# set_names = ['q07_1']

# run_names = ['q07r1','q07r2','q07r3','q07r4','q07r5','q07r6','q07r7','q07r8','q07r9']
# set_names = ['q07_1','q07_1','q07_1','q07_1','q07_1','q07_1','q07_1','q07_1','q07_1']

# run_names = ['q10r1','q10r2','q10r3','q10r4','q10r5','q10r6','q10r7','q10r8','q10r9']
# set_names = ['q10_2','q10_2','q10_2','q10_2','q10_2','q10_2','q10_2','q10_2','q10_2']

# run_names = ['q15r1','q15r2','q15r3','q15r4','q15r5','q15r6','q15r7','q15r8','q15r9']
# set_names = ['q15_2','q15_2','q15_2','q15_2','q15_2','q15_2','q15_2','q15_2','q15_2']

# run_names = ['q20r1','q20r2','q20r3','q20r4','q20r5','q20r6','q20r7','q20r8','q20r9']
# set_names = ['q20_2','q20_2','q20_2','q20_2','q20_2','q20_2','q20_2','q20_2','q20_2']

# run_names = ['q20r1']
# set_names = ['q20_2']

index = 0
nn=0
for run_name, set_name in zip(run_names, set_names):
    nn+=1
    print(set_name)
    print(run_name)
    print(index)
    
    print()


    ###########################################################################
    # IMPORT DoD STACK AND DoD BOOL STACK
    # DEFINE NAMES AND PATHS
    stack_name = 'DoD_stack' + '_' + set_name + '.npy' # Define stack name
    stack_bool_name = 'DoD_stack' + '_bool_' + set_name + '.npy' # Define stack bool name
    stack_path = os.path.join(DoDs_folder,stack_name) # Define stack path
    stack_bool_path = os.path.join(DoDs_folder,stack_bool_name) # Define stack bool path
    # LOAD THE STACK 
    stack = np.load(stack_path) # Load DoDs stack
    stack_bool = np.load(stack_bool_path) # Load DoDs bool stack
    ###########################################################################
    
    # DEFINE THE STACK DIMENSIONS
    dim_t, dim_y, dim_x, dim_delta = stack.shape # Define time dimension, crosswise dimension and longitudinal dimension
    

    # IMPORT DATA:
    
    # Import the background image (at this stage 6591x701)
    # channel_path = '/home/erri/Documents/PhD/Research/5_research_repos/PiQs_analysis/Photos/'+ run_name+ '/Img0001.jpg' # Import the background images
    # channel_rsz = Image.open(channel_path).resize((1260, 140), Image.LANCZOS)
    
    # Import the Bedload active area (BAA) envelope
    env_BAA_path = '/home/erri/Documents/PhD/Research/5_research_repos/PiQs_analysis/Maps/'+run_name+'/'+run_name+'_envBAA_map_activ_history.tiff'
    envBAA = Image.open(env_BAA_path)
    envBAA = np.array(envBAA)
    
    # Resizing:
    # Order has to be 0 to avoid negative numbers in the cumulate intensity.
    resampling_factor = 5
    envBAA_rsz = scipy.ndimage.zoom(envBAA, 1/resampling_factor, mode='nearest', order=1) 
    # envBAA_rsz = np.repeat(envBAA_rsz, 10, axis=1) # Rescale the DEM (dx/dy = 10)
    
    
    envBAA_bool_rsz = np.where(envBAA_rsz>1,1,0).astype(np.uint8) # This trims all the BAA area that are active 1 time and convert the matrix in bool
    # envBAA_bool_rsz = np.repeat(envBAA_rsz, 10, axis=1) # Rescale the DEM (dx/dy = 10)
    
    # Import the Bedload Active Area (MAA) map envelope as the sum of the intensiti value
    envBAA_act_cumulative_path = os.path.join('/home/erri/Documents/PhD/Research/5_research_repos/PiQs_analysis/Maps/', run_name, run_name + '_envBAA_act_cumulative.tiff')
    
    # TODO old version
    # envBAA_act_cumulative_rsz = Image.open(envBAA_act_cumulative_path).resize((126, 140), Image.LANCZOS)
    # envBAA_act_cumulative_rsz = np.array(envBAA_act_cumulative_rsz)
    # envBAA_act_cumulative_rsz = np.repeat(envBAA_act_cumulative_rsz, 10, axis=1) # Rescale the DEM (dx/dy = 10)
    
    env_BAA_act_cumulative = Image.open(envBAA_act_cumulative_path)
    env_BAA_act_cumulative = np.array(env_BAA_act_cumulative)
    
    # Resizing:
    # Order has to be 0 to avoid negative numbers in the cumulate intensity.
    resampling_factor = 5
    envBAA_act_cumulative_rsz = scipy.ndimage.zoom(env_BAA_act_cumulative, 1/resampling_factor, mode='nearest', order=1) 
    envBAA_act_cumulative_rsz = np.repeat(envBAA_act_cumulative_rsz, 10, axis=1) # Rescale the DEM (dx/dy = 10)
    
    
    # Import the laser survey mask
    # array mask for filtering data outside the channel domain
    # Different mask will be applied depending on the run due to different ScanArea
    # used during the laser surveys
    runs_list = ['q10_1', 'q10_2', 'q15_1', 'q20_1', 'q20_2'] # Old runs with old ScanArea
    mask_arr_name, mask_arr_path = 'array_mask.txt', home_dir # Mask for runs 07 onwards

    if run_name in runs_list:
        mask_arr_name, mask_arr_path = 'array_mask_0.txt', home_dir
        
    # Load mask
    mask_arr = np.loadtxt(os.path.join(mask_arr_path, mask_arr_name))
    mask_arr_rsz = np.where(mask_arr==-999, np.nan, 1) # Convert in mask with 0 and 1
    mask_arr_rsz = np.repeat(mask_arr_rsz, 10, axis=1) # Rescale the envMAA (dx/dy = 10) (at this stage 144x2790)
    

    # Import the Morphological active area (MAA) from DoD
    envMAA = stack[index,:,:, 0] # (at this stage 144x279)
    envMAA_arr_plot = np.where(np.isnan(envMAA), 0, envMAA)
    envMAA_rsz = np.repeat(envMAA_arr_plot, 10, axis=1) # Rescale the envMAA (dx/dy = 10) (at this stage 144x2790)
    
    
    
    envMAA_bool = stack_bool[index,:,:, 0]
    envMAA_arr_bool_plot = np.where(np.isnan(envMAA_bool), 0, envMAA_bool)
    envMAA_bool_rsz = np.repeat(envMAA_arr_bool_plot, 10, axis=1) # Rescale the envMAA (dx/dy = 10) (at this stage 144x2790)
    
    # Import the DEM
    DEM = np.loadtxt('/home/erri/Documents/PhD/Research/5_research_repos/DoD_analysis/surveys/q07_1/matrix_bed_norm_q07_1s0.txt', skiprows=8)
    DEM = np.where(DEM==-999, np.nan, DEM) # (at this stage 178x278)
    DEM_rsz = np.repeat(DEM, 10, axis=1) # Rescale the DEM (dx/dy = 10) (at this stage 144x2790)


    if set_name == 'q07_1':
        # Define the transformation parameters
        scale = 1.0 # Enlargement scale
        dx = 0 # Shift in x direction
        dy = 10 # Shift in y direction
        rot_angle = -0.55

    if set_name == 'q10_2':
        # Define the transformation parameters
        scale = 1.0 # Enlargement scale
        dx = 0 # Shift in x direction
        dy = 8 # Shift in y direction
        rot_angle = -0.3

    if set_name == 'q15_2':
        # Define the transformation parameters
        scale = 1.0 # Enlargement scale
        dx = 0 # Shift in x direction
        dy = 8 # Shift in y direction
        rot_angle = -0.4

    if set_name == 'q20_2':
        # Define the transformation parameters
        scale = 1.0 # Enlargement scale
        dx = -10 # Shift in x direction
        dy =8 # Shift in y direction
        rot_angle = -0.3
    
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

    envBAA_rsz_rsc = img_scaling_to_DEM(envBAA_rsz, scale, dx, dy, rot_angle) # envBAA resized, rescaled and rotated
    envBAA_bool_rsz_rsc = img_scaling_to_DEM(envBAA_bool_rsz, scale, dx, dy, rot_angle) # envBAA resized, rescaled and rotated
    envBAA_act_cumulative_rsz_rsc = img_scaling_to_DEM(envBAA_act_cumulative_rsz, scale, dx, dy, rot_angle) # envBAA resized, rescaled and rotated
    


    # CUT LASER OUTPUT TO FIT PHOTOS
    envMAA_rsz = envMAA_rsz[:, envMAA_rsz.shape[1]-1229:]
    envMAA_bool_rsz = envMAA_bool_rsz[:, envMAA_bool_rsz.shape[1]-1229:]
    DEM_rsz = DEM_rsz[:, DEM_rsz.shape[1]-1229:]
    mask_arr_rsz = mask_arr_rsz[:, mask_arr_rsz.shape[1]-1229:]
    
    DEM_rsz = DEM_rsz*mask_arr_rsz # Apply mask
    
    '''
    Preliminary output to overlap anf study corrispondences between maps
    envMAA_bool_rsz
    envBAA_bool_rsz_rsc
    envBAA_rsz_rsc
    DEM_rsz
    '''
    # PLOT MAPS OVERLAPPED
    # Convert matrices to images
    # channel_background = plt.imshow(channel_rsz_rsc, alpha=1.0)
    envBAA = plt.imshow(envBAA_bool_rsz_rsc, cmap='Greens', alpha=0.4, vmin=0, vmax=1)
    envMAA_bool = plt.imshow(np.where(envMAA_bool_rsz == 0, np.nan, envMAA_bool_rsz), cmap='RdBu', origin='upper', alpha=0.5, interpolation_stage='rgba')
    # DEM = plt.imshow(np.where(np.isnan(DEM_rsz), np.nan, DEM_rsz), cmap='gist_earth', origin='upper', alpha=1.0, interpolation_stage='rgba', vmin=-20, vmax=20)
    # Set title and show the plot
    plt.title(run_name)
    plt.axis('off')
    plt.savefig(os.path.join(report_dir, set_name,
                run_name + '_BAA_MAA.pdf'), dpi=600)

    if index == 0:
        plt.savefig(os.path.join(report_dir, set_name,
                    set_name + '_report_BAA_MAA.pdf'), dpi=1200)

    if nn == 1:
        plt.savefig(os.path.join(report_dir, 'report_BAA_MAA.pdf'), dpi=1200)

    plt.show()

    if len(run_names)>1:
        if index > 0:
            merger = PyPDF2.PdfMerger()
    
            # Open and append the existing PDF
            with open(os.path.join(report_dir, set_name, set_name + '_report_BAA_MAA.pdf'), "rb") as existing_file:
                merger.append(existing_file)
    
            # Open and append the new PDF chart
            with open(os.path.join(report_dir, set_name, run_name + '_BAA_MAA.pdf'), "rb") as chart_file:
                merger.append(chart_file)
    
            # Save the merged PDF
            with open(os.path.join(report_dir, set_name, set_name + '_report_BAA_MAA.pdf'), "wb") as merged_file:
                merger.write(merged_file)
            
        if set_names[index]==set_names[index+1]:
            index+=1
        elif set_names[index]!=set_names[index+1]:
            index=0
        else:
            pass
                
        if nn>1:
            merger = PyPDF2.PdfMerger()
    
            # Open and append the existing PDF
            with open(os.path.join(report_dir, 'report_BAA_MAA.pdf'), "rb") as existing_file:
                merger.append(existing_file)
    
            # Open and append the new PDF chart
            with open(os.path.join(report_dir, set_name, run_name + '_BAA_MAA.pdf'), "rb") as chart_file:
                merger.append(chart_file)
    
            # Save the merged PDF
            with open(os.path.join(report_dir, 'report_BAA_MAA.pdf'), "wb") as merged_file:
                merger.write(merged_file)
    

#%%
    # Correlation analysis between BAA, DoD and DEM
    '''
    Data available:
        envMAA_rsz
        envBAA_rsz_rsc
        DEM_rsz
    '''
    # RESIZE ARRAYS:
    # Cut laser surveys rows
    envMAA_rsz = envMAA_rsz[:envBAA_rsz_rsc.shape[0],:]
    envMAA_bool_rsz = envMAA_bool_rsz[:envBAA_rsz_rsc.shape[0],:]
    DEM_rsz = DEM_rsz[:DEM_rsz.shape[0],:]
    
    # Cut images columns
    envBAA_rsz_rsc = envBAA_rsz_rsc[:,:envMAA_rsz.shape[1]]
    envBAA_bool_rsz_rsc = envBAA_bool_rsz_rsc[:,:envMAA_rsz.shape[1]]
    envBAA_act_cumulative_rsz_rsc = envBAA_act_cumulative_rsz_rsc[:,:envMAA_rsz.shape[1]]
    
    # With BAA cumulative map
    MAA_BAA_stack = np.stack((envMAA_rsz,envBAA_act_cumulative_rsz_rsc))
    
    # # With BAA number of ative times
    # MAA_BAA_stack = np.stack((envMAA_rsz,envBAA_rsz_rsc))
    
    
    # Unroll the stack and define a new array where the first row is MAA and
    # the second is BAA
    num_rows, num_slices, slice_length = MAA_BAA_stack.shape
    MAA_BAA_dataset = MAA_BAA_stack.reshape(num_rows, num_slices*slice_length)
    
    # Trim column where BAA is zero
    MAA_BAA_dataset_check = MAA_BAA_dataset[1] != 0
    MAA_BAA_dataset = MAA_BAA_dataset[:, MAA_BAA_dataset_check]
    
    if set_name == 'q07_1':
        c=4
    if set_name == 'q10_2':
        c=2
    if set_name == 'q15_2':
        c=2/3
    if set_name == 'q20_2':
        c=1
        
        
    # Create the scatter plot
    plt.scatter(MAA_BAA_dataset[0,:], MAA_BAA_dataset[1,:]/c, s=2, c='red', marker='o')
    '''
    Considering the cumulate values of the bedload activity could lead to a
    chart that is very difficult to read. In fact, given a costatn timespan
    between shoots the run duratio (that change a lot between runs) affects the
    envelope activity value.
    '''
    # Add labels and a title to the plot
    plt.xlabel('DoD value')
    plt.ylabel('Envelope cumulative value')
    plt.title(run_name + ' _ Scatter Plot')
    
    # Save image
    plt.savefig(os.path.join(report_dir, set_name, run_name + '_scatter_envBAA_cumulative_DoD'), dpi=400)
    
    # Show the plot
    plt.show()
    
    
    #%%
    '''
    Histograms with the distribution of frequency of the DoD and BAA cumulated values
    '''
    
    # MAA
    
    MAA_values = MAA_BAA_dataset[0,:]
    MAA_values = MAA_values[MAA_values != 0] # Trim zero values
    
    
    # Plot the histogram
    plt.hist(MAA_values, bins=np.arange(min(MAA_values), max(MAA_values) + 1), rwidth=0.6, align='left', density=True, alpha=0.6, label='Frequency')
    
    # Estimate the PDF using Kernel Density Estimation
    kde = gaussian_kde(MAA_values)
    x_vals = np.linspace(min(MAA_values), max(MAA_values), 1000)
    y_vals = kde(x_vals)
    
    # Plot the PDF on top of the histogram
    plt.plot(x_vals, y_vals, '-r', label='PDF')
    
    # Add labels and title
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.title('Histogram with PDF - MAA_values')
    
    # Show a legend with the PDF label
    plt.legend()
    
    # Display the plot
    plt.show()
    
    
    
    # BAA
    
    BAA_values = MAA_BAA_dataset[1,:]/c
    BAA_values = BAA_values[BAA_values > 0] # Trim zero values
    
    # Plot the histogram
    plt.hist(BAA_values, bins=np.arange(min(BAA_values), max(BAA_values) + 1), rwidth=0.6, align='left', density=True, alpha=0.6, label='Frequency')
    
    # Estimate the PDF using Kernel Density Estimation
    kde = gaussian_kde(BAA_values)
    x_vals = np.linspace(min(BAA_values), max(BAA_values), 1000)
    y_vals = kde(x_vals)
    
    # Plot the PDF on top of the histogram
    plt.plot(x_vals, y_vals, '-r', label='PDF')
    
    # Add labels and title
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.title('Histogram with PDF - Bedload intensity cumulated values')
    
    # Show a legend with the PDF label
    plt.legend()
    
    # Display the plot
    plt.show()
