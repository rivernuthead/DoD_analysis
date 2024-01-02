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
from scipy.ndimage import rotate
from PIL import Image
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import scipy.ndimage
import PyPDF2
from functions import *
    
# ---------------------------------------------------------------------------- #
#                                 FOLDER SETUP                                 #
# ---------------------------------------------------------------------------- #
home_dir = os.getcwd() # Home directory
report_dir = os.path.join(home_dir, 'output')
run_dir = os.path.join(home_dir, 'surveys')
DoDs_folder = os.path.join(home_dir, 'DoDs', 'DoDs_stack') # Input folder
PiQs_folder_path = '/home/erri/Documents/PhD/Research/5_research_repos/PiQs_analysis/'

# ---------------------------------------------------------------------------- #
#                                   RUN SETUP                                  #
# ---------------------------------------------------------------------------- #

analysis_list = [
    'correlation_analysis',
    'hist_DoD_BAA_cumul_values',
]


run_names = ['q20r1']
set_names = ['q20_2']


# ---------------------------------------------------------------------------- #
#                            LOOP ALL OVER THE RUNS                            #
# ---------------------------------------------------------------------------- #

index = 0
nn=0
for run_name, set_name in zip(run_names, set_names):
    nn+=1
    run=run_name
    print(set_name)
    print(run_name)
    print(index)
    
    print()


    # ---------------------------------------------------------------------------- #
    #                                  IMPORT DATA                                 #
    # ---------------------------------------------------------------------------- #
    
    # -------------------- IMPORT DoD STACK AND DoD BOOL STACK ------------------- #
    # DEFINE PATHS
    DoD_stack_name = 'DoD_stack' + '_' + set_name + '.npy' # Define stack name
    DoD_stack_bool_name = 'DoD_stack' + '_bool_' + set_name + '.npy' # Define stack bool name
    DoD_stack_path = os.path.join(DoDs_folder,DoD_stack_name) # Define stack path
    DoD_stack_bool_path = os.path.join(DoDs_folder,DoD_stack_bool_name) # Define stack bool path
    # LOAD STACKS
    DoD_stack = np.load(DoD_stack_path) # Load DoDs stack
    DoD_stack_bool = np.load(DoD_stack_bool_path)  # Load DoDs bool stack
    
    # To obtain the actual pixel resolution (5x50) the DoD columns need to be repeted
    DoD_stack_bool_rsz = np.repeat(DoD_stack_bool, 10, axis=2)
    
    # Get the stack dimension
    # Define time dimension, crosswise dimension and longitudinal dimension
    DoD_dim_t, DoD_dim_y, DoD_dim_x, DoD_dim_delta = DoD_stack.shape
    # ---------------------------------------------------------------------------- #


    # --------------------------- IMPORT BAA STACK --------------------------- #
    '''BAA stack is the binary file in which the activity maps are stored '''
    stack_BAA_path = os.path.join(PiQs_folder_path, 'activity_stack/activity_stack_cleaned',run_name + '_BAA_stack_LR5_cld.npy') # Set the cleaned stack path
    stack_BAA = np.load(stack_BAA_path) # Load the stack
    
    # Get the stack dimension
    # Define time dimension, crosswise dimension and longitudinal dimension
    photo_dim_t, photo_dim_y, photo_dim_x = stack_BAA.shape
    
    # Compute the cumulate active times envelope
    env_BAA = np.nansum(stack_BAA, axis=0)
    env_BAA = np.array(env_BAA, dtype=np.uint8)
    
    # ---------------------------------------------------------------------------- #
#%%


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
    
    # Convert envBAA in image:
    env_BAA_img = Image.fromarray(env_BAA)
    # env_BAA_img.show()
    env_BAA_rsz = img_scaling_to_DEM(env_BAA, scale, dx, dy, rot_angle) # envBAA resized, rescaled and rotated
    # Save image
    env_BAA_rsz.save("output_BAA.png")

    # ------------------------- IMPORT LASER SURVEY MASK ------------------------- #
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
    # ---------------------------------------------------------------------------- #



    # Convert mask in image:
    mask_arr_rsz_img = Image.fromarray(np.array(mask_arr_rsz,dtype=np.uint8))
    # Save image
    mask_arr_rsz_img.save("output_mask.png")
    
    
    
#%%

    # ---------------------------- IMPORT MAA FROM DOD --------------------------- #
    envMAA = DoD_stack[index, :, :, 0]  # (at this stage 144x279)
    envMAA_arr_plot = np.where(np.isnan(envMAA), 0, envMAA)
    envMAA_rsz = np.repeat(envMAA_arr_plot, 10, axis=1) # Rescale the envMAA (dx/dy = 10) (at this stage 144x2790)
    
    
    
    envMAA_bool = DoD_stack_bool[index,:,:, 0]
    envMAA_arr_bool_plot = np.where(np.isnan(envMAA_bool), 0, envMAA_bool)
    envMAA_bool_rsz = np.repeat(envMAA_arr_bool_plot, 10, axis=1) # Rescale the envMAA (dx/dy = 10) (at this stage 144x2790)
    
    # Import the DEM
    DEM = np.loadtxt('/home/erri/Documents/PhD/Research/5_research_repos/DoD_analysis/surveys/q07_1/matrix_bed_norm_q07_1s0.txt', skiprows=8)
    DEM = np.where(DEM==-999, np.nan, DEM) # (at this stage 178x278)
    DEM_rsz = np.repeat(DEM, 10, axis=1) # Rescale the DEM (dx/dy = 10) (at this stage 144x2790)


#%%

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
    
    
    
    envBAA_rsz = envBAA_rsz.astype(np.uint16)
    envBAA_bool_rsz = envBAA_bool_rsz.astype(np.uint16)
    envBAA_act_cumulative_rsz = envBAA_act_cumulative_rsz.astype(np.uint16)
    # activated_pixel_envelope_rsz = activated_pixel_envelope_rsz.astype(np.uint16)
    # deactivated_pixel_envelope_rsz = deactivated_pixel_envelope_rsz.astype(np.uint16)
    
    
    envBAA_rsz_rsc = img_scaling_to_DEM(envBAA_rsz, scale, dx, dy, rot_angle) # envBAA resized, rescaled and rotated
    envBAA_bool_rsz_rsc = img_scaling_to_DEM(envBAA_bool_rsz, scale, dx, dy, rot_angle) # envBAA resized, rescaled and rotated
    envBAA_act_cumulative_rsz_rsc = img_scaling_to_DEM(envBAA_act_cumulative_rsz, scale, dx, dy, rot_angle) # envBAA resized, rescaled and rotated
    # TODO add Activated and Deactivated data:
    # activated_pixel_envelope_rsz_rsc = img_scaling_to_DEM(activated_pixel_envelope_rsz, scale, dx, dy, rot_angle)
    # deactivated_pixel_envelope_rsz_rsc = img_scaling_to_DEM(deactivated_pixel_envelope_rsz, scale, dx, dy, rot_angle)



    # ---------------------- CUT LASER OUTPUT TO FIT PHOTOS ---------------------- #
    envMAA_rsz = envMAA_rsz[:, envMAA_rsz.shape[1]-1229:]
    envMAA_bool_rsz = envMAA_bool_rsz[:, envMAA_bool_rsz.shape[1]-1229:]
    DEM_rsz = DEM_rsz[:, DEM_rsz.shape[1]-1229:]
    mask_arr_rsz = mask_arr_rsz[:, mask_arr_rsz.shape[1]-1229:]
    
    DEM_rsz = DEM_rsz*mask_arr_rsz # Apply mask
    # ---------------------------------------------------------------------------- #

    '''
    Preliminary output to overlap anf study correspondences between maps
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
    # TODO plot Activated and Deactivated maps
    # DEM = plt.imshow(np.where(np.isnan(DEM_rsz), np.nan, DEM_rsz), cmap='gist_earth', origin='upper', alpha=1.0, interpolation_stage='rgba', vmin=-20, vmax=20)
    # Set title and show the plot
    plt.title(run_name)
    plt.axis('off')
    plt.savefig(os.path.join(report_dir, set_name,run_name + '_BAA_MAA.pdf'), dpi=600)

    if index == 0:
        plt.savefig(os.path.join(report_dir, set_name,set_name + '_report_BAA_MAA.pdf'), dpi=1200)

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