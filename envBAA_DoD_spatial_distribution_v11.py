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
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import scipy.ndimage
import PyPDF2
from functions import *
    
# ---------------------------------------------------------------------------- #
#                                 FOLDER SETUP                                 #
# ---------------------------------------------------------------------------- #
home_dir = os.getcwd() # Home directory

run_dir = os.path.join(home_dir, 'surveys')
DoDs_folder = os.path.join(home_dir,'output', 'DoDs', 'DoDs_stack') # Input folder
PiQs_folder_path = '/home/erri/Documents/PhD/Research/5_research_repos/PiQs_analysis/'

# ---------------------------------------------------------------------------- #
#                                   RUN SETUP                                  #
# ---------------------------------------------------------------------------- #

analysis_list = [
    'correlation_analysis',
    'hist_DoD_BAA_cumul_values',
]

# run_names = ['q07r1','q07r2','q07r3','q07r4','q07r5','q07r6','q07r7','q07r8','q07r9'
#         ,'q10r1','q10r2','q10r3','q10r4','q10r5','q10r6','q10r7','q10r8','q10r9'
#         ,'q15r1','q15r2','q15r3','q15r4','q15r5','q15r6','q15r7','q15r8','q15r9'
#         ,'q20r1','q20r2','q20r3','q20r4','q20r5','q20r6','q20r7','q20r8','q20r9']
# set_names = ['q07_1','q07_1','q07_1','q07_1','q07_1','q07_1','q07_1','q07_1','q07_1'
#         ,'q10_2','q10_2','q10_2','q10_2','q10_2','q10_2','q10_2','q10_2','q10_2'
#         ,'q15_2','q15_2','q15_2','q15_2','q15_2','q15_2','q15_2','q15_2','q15_2'
#         ,'q20_2','q20_2','q20_2','q20_2','q20_2','q20_2','q20_2','q20_2','q20_2']


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

    # ---------------------- DEFINE THE INPUT DATA TIMESCALE --------------------- #
    # env_tscale_array = [5, 5, 5, 5]
    env_tscale_array = [1,1,1,1]
    # env_tscale_array = [5,7,12,18]
    # env_tscale_array = [10,14,24,36]
    # ---------------------------- ENVELOPE TIMESCALE ---------------------------- #
    if run_name[1:3] == '07':
        env_tscale = env_tscale_array[3]
    if run_name[1:3] == '10':
        env_tscale = env_tscale_array[2]
    if run_name[1:3] == '15':
        env_tscale = env_tscale_array[1]
    if run_name[1:3] == '20':
        env_tscale = env_tscale_array[0]


    # ---------------------------------------------------------------------------- #
    #                                  IMPORT DATA                                 #
    # ---------------------------------------------------------------------------- #
    # -------------------- IMPORT DoD STACK AND DoD BOOL STACK ------------------- #
    # DEFINE PATHS
    report_dir = os.path.join(home_dir, 'output', 'report_'+run)
    DoD_stack_name = 'DoD_stack' + '_' + set_name + '.npy' # Define stack name
    DoD_stack_bool_name = 'DoD_stack' + '_bool_' + set_name + '.npy' # Define stack bool name
    DoD_stack_path = os.path.join(DoDs_folder,DoD_stack_name) # Define stack path
    DoD_stack_bool_path = os.path.join(DoDs_folder,DoD_stack_bool_name) # Define stack bool path
    # LOAD STACKS
    DoD_stack = np.load(DoD_stack_path) # Load DoDs stack
    DoD_stack_bool = np.load(DoD_stack_bool_path)  # Load DoDs bool stack

    '''
    DoD input stack structure:
        
        DoD_stack[time,y,x,delta]
        DoD_stack_bool[time,y,x,delta]
        
        - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - >    delta
        |  DoD 1-0  DoD 2-0  DoD 3-0  DoD 4-0  DoD 5-0  DoD 6-0  DoD 7-0  DoD 8-0  DoD 9-0
        |  DoD 2-1  DoD 3-1  DoD 4-1  DoD 5-1  DoD 6-1  DoD 7-1  DoD 8-1  DoD 9-1
        |  DoD 3-2  DoD 4-2  DoD 5-2  DoD 6-2  DoD 7-2  DoD 8-2  DoD 9-2
        |  DoD 4-3  DoD 5-3  DoD 6-3  DoD 7-3  DoD 8-3  DoD 9-3
        |  DoD 5-4  DoD 6-4  DoD 7-4  DoD 8-4  DoD 9-4
        |  DoD 6-5  DoD 7-5  DoD 8-5  DoD 9-5
        |  DoD 7-6  DoD 8-6  DoD 9-6
        |  DoD 8-7  DoD 9-7
        |  DoD 9-8
        |
        v
        time
            
    '''
    # ---------------------------------------------------------------------------- #


    # Define time dimension, crosswise dimension and longitudinal dimension
    dim_t, dim_y, dim_x, dim_delta = DoD_stack.shape
    
    # --------------------------- IMPORT BAA ENVELOPES --------------------------- #
    '''BAA envelopes where in each cell the number of time it has been active is stored '''
    env_BAA_path = os.path.join(PiQs_folder_path, 'Maps',run_name, run_name+'_envBAA_map_activ_history.tiff')
    envBAA = Image.open(env_BAA_path)
    envBAA = np.array(envBAA)
    
    # Resizing:
    # Order has to be 0 to avoid negative numbers in the cumulate intensity.
    resampling_factor = 5
    envBAA_rsz = scipy.ndimage.zoom(envBAA, 1/resampling_factor, mode='nearest', order=1) 
    # envBAA_rsz = np.repeat(envBAA_rsz, 10, axis=1) # Rescale the DEM (dx/dy = 10)
    
    envBAA_bool_rsz = np.where(envBAA_rsz>1,1,0).astype(np.uint8) # This trims all the BAA area that are active 1 time and convert the matrix in bool
    # envBAA_bool_rsz = np.repeat(envBAA_rsz, 10, axis=1) # Rescale the DEM (dx/dy = 10)
    # ---------------------------------------------------------------------------- #

    
    # ------------- IMPORT BAA ENVELOPES AS CUMULATIVE INTENSITY MAPS ------------ #
    '''BAA envelopes where in each cell the cumulative sum of bedload intensity is stored '''
    envBAA_act_cumulative_path = os.path.join(PiQs_folder_path, 'Maps', run_name, run_name + '_envBAA_act_cumulative.tiff')
    
    env_BAA_act_cumulative = Image.open(envBAA_act_cumulative_path)
    env_BAA_act_cumulative = np.array(env_BAA_act_cumulative)
    
    # Resizing:
    # Order has to be 0 to avoid negative numbers in the cumulate intensity.
    resampling_factor = 5
    envBAA_act_cumulative_rsz = scipy.ndimage.zoom(env_BAA_act_cumulative, 1/resampling_factor, mode='nearest', order=1) 
    envBAA_act_cumulative_rsz = np.repeat(envBAA_act_cumulative_rsz, 10, axis=1) # Rescale the DEM (dx/dy = 10)
    # ---------------------------------------------------------------------------- #


    # -------------- IMPORT MAPS OF ACTIVATED AND DEACTIVATED AREAS -------------- #
    # # TODO COMPLETE THIS
    # '''The stack bool diff comes from the difference of maps taken at a given timescale'''
    # path_partial_envelopes = os.path.join(PiQs_folder_path,'activity_stack/activity_stack_cleaned/partial_envelopes', run + '_envTscale' + str(env_tscale))
    # stack_bool_diff_cld_path = os.path.join(path_partial_envelopes, run+ '_envT' + str(env_tscale) + '_partial_stack_bool_diff_cld.npy')
    # stack_bool_diff_cld = np.load(stack_bool_diff_cld_path)

    # activated_pixel_envelope   = np.nansum(stack_bool_diff_cld==+1, axis=0)
    # deactivated_pixel_envelope = np.nansum(stack_bool_diff_cld==-1, axis=0)

    # # Resizing:
    # # Order has to be 0 to avoid negative numbers in the cumulate intensity.
    # resampling_factor = 5
    # activated_pixel_envelope_rsz = scipy.ndimage.zoom(activated_pixel_envelope, 1/resampling_factor, mode='nearest', order=1) 
    # activated_pixel_envelope_rsz = np.repeat(activated_pixel_envelope_rsz, 10, axis=1) # Rescale the DEM (dx/dy = 10)
    # deactivated_pixel_envelope_rsz = scipy.ndimage.zoom(deactivated_pixel_envelope, 1/resampling_factor, mode='nearest', order=1) 
    # deactivated_pixel_envelope_rsz = np.repeat(deactivated_pixel_envelope_rsz, 10, axis=1) # Rescale the DEM (dx/dy = 10)
    # ---------------------------------------------------------------------------- #



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
        scale = 0.2 # Enlargement scale
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
    # plt.savefig(os.path.join(report_dir, set_name,run_name + '_BAA_MAA.pdf'), dpi=600)

    # if index == 0:
    #     plt.savefig(os.path.join(report_dir, set_name,set_name + '_report_BAA_MAA.pdf'), dpi=1200)

    # if nn == 1:
    #     plt.savefig(os.path.join(report_dir, 'report_BAA_MAA.pdf'), dpi=1200)

    plt.show()


    if 'correlation_analysis' in analysis_list:
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
        # plt.savefig(os.path.join(report_dir, set_name, run_name + '_scatter_envBAA_cumulative_DoD'), dpi=400)
        
        # Show the plot
        plt.show()
    
    
    if 'hist_DoD_BAA_cumul_values' in analysis_list:
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
