#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 30 14:11:51 2022

@author: erri

The aims of this script are:
    1. 

INPUT (as .npy binary files):
    DoD_stack1 : 3D numpy array stack
        Stack on which DoDs are stored as they are, with np.nan
    DoD_stack1_bool : 3D numpy array stack
        Stack on which DoDs are stored as -1, 0, +1 data, also with np.nan
OUTPUTS:
    For each run the scrip gives you a map where the envelope of the Bedload
    Active Area and the corresponding DoD are overlapped
    
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

#%%############################################################################
# VERY FIRST SETUP
start = time.time() # Set initial time


# SINGLE RUN NAME
runs = ['q20_2']
# runs = ['q07_1', 'q10_2', 'q10_3', 'q15_2', 'q15_3', 'q20_2']

for run in runs:
    
    # ALENGTH OF THE ANALYSIS WINDOW IN NUMBER OF CELL
    analysis_window = 123 # Number of columns compared to the PiQs photo length
    
    # FOLDER SETUP
    home_dir = os.getcwd() # Home directory
    report_dir = os.path.join(home_dir, 'output')
    run_dir = os.path.join(home_dir, 'surveys')
    DoDs_folder = os.path.join(home_dir, 'DoDs', 'DoDs_stack') # Input folder
    
    if not(os.path.exists(os.path.join(report_dir, run,  'stack_analysis'))):
        os.mkdir(os.path.join(report_dir, run,  'stack_analysis'))
    
    
    ###############################################################################
    # IMPORT DoD STACK AND DoD BOOL STACK
    DoDs_folder = os.path.join(home_dir, 'DoDs', 'DoDs_stack') # Input folder
    stack_name = 'DoD_stack' + '_' + run + '.npy' # Define stack name
    stack_bool_name = 'DoD_stack' + '_bool_' + run + '.npy' # Define stack bool name
    stack_path = os.path.join(DoDs_folder,stack_name) # Define stack path
    stack_bool_path = os.path.join(DoDs_folder,stack_bool_name) # Define stack bool path
    
    stack = np.load(stack_path) # Load DoDs stack
    stack_bool = np.load(stack_bool_path) # Load DoDs boolean stack
    
    dim_t, dim_y, dim_x, dim_delta = stack.shape # Define time dimension, crosswise dimension and longitudinal dimension
    
    
    '''
    Data Structure
    
    DoD1-0  DoD2-0  DoD3-0  DoD4-0  DoD5-0  DoD6-0  DoD7-0  DoD8-0  DoD9-0
    DoD2-1  DoD3-1  DoD4-1  DoD5-1  DoD6-1  DoD7-1  DoD8-1  DoD9-1
    DoD3-2  DoD4-2  DoD5-2  DoD6-2  DoD7-2  DoD8-2  DoD9-2
    DoD4-3  DoD5-3  DoD6-3  DoD7-3  DoD8-3  DoD9-3
    DoD5-4  DoD6-4  DoD7-4  DoD8-4  DoD9-4
    DoD6-5  DoD7-5  DoD8-5  DoD9-5
    DoD7-6  DoD8-6  DoD9-6
    DoD8-7  DoD9-7
    DoD9-8
    
    stack = [h,:,:, delta]
    
    '''
    
    
    
    #%%############################################################################
    '''
    DoD envelope
    '''
    # 1 timespan envelope
    for i in range(0,stack.shape[0]):
        envMAW_arr = np.nansum(np.abs(stack_bool[:stack.shape[0]-i,:,dim_x-analysis_window:,i]), axis=0) 
        envMAW_arr = np.repeat(envMAW_arr, 10, axis=1)
        
        envMAW = Image.fromarray(np.array(envMAW_arr).astype(np.uint8))
        envMAW.save(os.path.join(report_dir, run, run + '_envMAW_' + str(i+1) + 'tsp.tiff'))
        
        envMAW_bool_arr = np.where(envMAW_arr>0, 1, 0)
        envMAW_bool = Image.fromarray(np.array(envMAW_bool_arr).astype(np.uint8))
        envMAW_bool.save(os.path.join(report_dir, run, run + '_envMAW_bool_' + str(i+1) + 'tsp.tiff'))
        
        


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
    channel = Image.open('/home/erri/Documents/PhD/Research/5_research_repos/PiQs_analysis/Photos/'+ run_name+ '/Img0001.jpg') # Import the background images
    
    # Import the Bedload active area (BAA) envelope
    BAA_path = '/home/erri/Documents/PhD/Research/5_research_repos/PiQs_analysis/Maps/'+run_name+'/'+run_name+'_envBAA_map_activ_history.tiff'
    # envBAA = cv2.imread(BAA_path)
    # envBAA = envBAA[:700,:6290,2] # Select the active band (at this stage 700x6290)
    # envBAA_bool = np.where(envBAA>1, 1, 0)
    
    envBAA_rsz = Image.open(BAA_path).resize((126, 140), Image.LANCZOS)
    envBAA_rsz = np.array(envBAA_rsz)
    envBAA_rsz = np.where(envBAA_rsz>1,1,0).astype(np.uint8)
    envBAA_rsz = np.repeat(envBAA_rsz, 10, axis=1) # Rescale the DEM (dx/dy = 10)
    
    # Import the Morphological active area (MAA) from DoD
    envMAA = stack[index,:,:, 0] # (at this stage 144x279)
    envMAA = stack_bool[index,:,:, 0]
    envMAA_arr_plot = np.where(np.isnan(envMAA), 0, envMAA)
    # envMAA_arr_plot = envMAA_arr_plot[8:,:]
    envMAA_rsz = np.repeat(envMAA_arr_plot, 10, axis=1) # Rescale the envMAA (dx/dy = 10) (at this stage 144x2790)
    
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
    
    # CUT LASER OUTPUT TO FIT PHOTOS
    envMAA_rsz = envMAA_rsz[:,envMAA_rsz.shape[1]-1229:]
    DEM_rsz    = DEM_rsz[:,DEM_rsz.shape[1]-1229:]
    

    # Convert matrices to images
    # img1 = plt.imshow(channel, alpha=1.0)
    img2 = plt.imshow(envBAA_rsz_rsc, cmap='Greens', alpha=0.4, vmin=0, vmax=1)
    img3 = plt.imshow(np.where(envMAA_rsz ==0, np.nan, envMAA_rsz), cmap='RdBu', origin='upper', alpha=0.5, interpolation_stage='rgba')
    # img3 = plt.imshow(np.where(np.isnan(DEM), 100, np.nan), cmap='bone', origin='upper', alpha=1.0, interpolation_stage='rgba')
    # Set title and show the plot
    plt.title(run_name)
    plt.axis('off')
    plt.savefig(os.path.join(report_dir, set_name, run_name + '_BAA_MAA.pdf'), dpi=600 )
    
    if index==0:
        plt.savefig(os.path.join(report_dir, set_name, set_name + '_report_BAA_MAA.pdf'), dpi=1200 )
        
    if nn==1:
        plt.savefig(os.path.join(report_dir, 'report_BAA_MAA.pdf'), dpi=1200 )
    
    plt.show()
    
    if index>0:
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