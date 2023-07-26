#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 30 14:11:51 2022

@author: erri

Pixel age analysis over stack DoDs

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
import matplotlib.pyplot as plt
import PyPDF2
from PyPDF2 import PdfFileMerger, PdfFileReader, PdfFileWriter

#%%############################################################################
# VERY FIRST SETUP
start = time.time() # Set initial time


# SINGLE RUN NAME
# runs = ['q20_2']
runs = ['q07_1', 'q10_2', 'q10_3', 'q15_2', 'q15_3', 'q20_2']

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
    
    
    # print('###############' +'\n#    ' + run + '\n#' + '##############')
    
    # ###############################################################################
    # # IMPORT RUN PARAMETERS from file parameters.txt
    # # variable run must be as 'q' + discharge + '_' repetition number
    # # Parameters.txt structure:
    # # discharge [l/s],repetition,run time [min],Texner discretization [-], Channel width [m], slope [m/m]
    # # Load parameter matrix:
    # parameters = np.loadtxt(os.path.join(home_dir, 'parameters.txt'),
    #                         delimiter=',',
    #                         skiprows=1)
    # # Extract run parameter depending by run name
    # run_param = parameters[np.intersect1d(np.argwhere(parameters[:,1]==float(run[-1:])),np.argwhere(parameters[:,0]==float(run[1:3])/10)),:]
    
    # # Run time data
    # dt = run_param[0,2] # dt between runs [min] (real time)
    # dt_xnr = run_param[0,3] # temporal discretization in terms of Exner time (Texner between runs)
    
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

# run_names = ['q07r1','q07r2','q07r3','q07r4','q07r5','q07r6','q07r7','q07r8','q07r9'
#         ,'q10r1','q10r2','q10r3','q10r4','q10r5','q10r6','q10r7','q10r8','q10r9'
#         ,'q15r1','q15r2','q15r3','q15r4','q15r5','q15r6','q15r7','q15r8','q15r9'
#         ,'q20r1','q20r2','q20r3','q20r4','q20r5','q20r6','q20r7','q20r8','q20r9']
# set_names = ['q07_1','q07_1','q07_1','q07_1','q07_1','q07_1','q07_1','q07_1','q07_1'
#         ,'q10_2','q10_2','q10_2','q10_2','q10_2','q10_2','q10_2','q10_2','q10_2'
#         ,'q15_2','q15_2','q15_2','q15_2','q15_2','q15_2','q15_2','q15_2','q15_2'
#         ,'q20_2','q20_2','q20_2','q20_2','q20_2','q20_2','q20_2','q20_2','q20_2']
# run_names = ['q07r1']

# run_names = ['q20r1','q20r2','q20r3','q20r4','q20r5','q20r6','q20r7','q20r8','q20r9']
# set_names = ['q20_2','q20_2','q20_2','q20_2','q20_2','q20_2','q20_2','q20_2','q20_2']

# run_names = ['q07r1','q07r2','q07r3','q07r4','q07r5','q07r6','q07r7','q07r8','q07r9']
# set_names = ['q07_1','q07_1','q07_1','q07_1','q07_1','q07_1','q07_1','q07_1','q07_1']

run_names = ['q07r1']
set_names = ['q07_1']

index = 0
nn=0
for run_name, set_name in zip(run_names, set_names):
    nn+=1
    print(set_name)
    print(run_name)
    print(index)
    
    print()

    ###############################################################################
    # IMPORT DoD STACK AND DoD BOOL STACK
    stack_name = 'DoD_stack' + '_' + set_name + '.npy' # Define stack name
    stack_bool_name = 'DoD_stack' + '_bool_' + set_name + '.npy' # Define stack bool name
    stack_path = os.path.join(DoDs_folder,stack_name) # Define stack path
    stack_bool_path = os.path.join(DoDs_folder,stack_bool_name) # Define stack bool path
    
    stack = np.load(stack_path) # Load DoDs stack
    stack_bool = np.load(stack_bool_path) # Load DoDs boolean stack
    
    dim_t, dim_y, dim_x, dim_delta = stack.shape # Define time dimension, crosswise dimension and longitudinal dimension
    
    
    
        
    
    channel = Image.open('/home/erri/Documents/PhD/Research/5_research_repos/PiQs_analysis/Photos/'+ run_name+ '/Img0001.jpg') # Open image as image
    # diff_arr = np.array(diff) # Convert image as numpy array
    
    envBAW = cv2.imread('/home/erri/Documents/PhD/Research/5_research_repos/PiQs_analysis/Maps/'+run_name+'/'+run_name+'_envBAW.tiff')
    envBAW = envBAW[:700,:6290,2] # Select the active band
    
    # h, w = envBAW.shape[:2]
    # envBAW = cv2.resize(envBAW, (int(70), int(629)))
    
    
    envMAW = stack_bool[index,:,:, 0]
    envMAW_arr_plot = np.where(np.isnan(envMAW), 0, envMAW)
    # envMAW_arr_plot = envMAW_arr_plot[8:,:]
    
    DEM = np.loadtxt('/home/erri/Documents/PhD/Research/5_research_repos/DoD_analysis/surveys/q07_1/matrix_bed_norm_q07_1s0.txt', skiprows=8)
    DEM = np.where(DEM==-999, np.nan, DEM)
    # DEM = DEM[:,dim_x-analysis_window:]
    # def add_zeros_rows(matrix, num_rows):
    #     # Get the number of columns in the matrix
    #     num_cols = matrix.shape[1]
        
    #     # Create a new array of the appropriate size with NaN values
    #     nan_rows = np.empty((num_rows, num_cols))
    #     nan_rows[:] = 0
        
    #     # Stack the NaN rows onto the bottom of the matrix
    #     new_matrix = np.vstack([matrix, nan_rows])
        
    #     return new_matrix
    
    # envMAW_arr_plot = add_zeros_rows(envMAW_arr_plot, 100)
    
    
    # # envMAW_zoom = zoom(envMAW_arr_plot[6:,:], (envBAW.size[1]/envMAW_arr_plot.shape[0], envBAW.size[0]/envMAW_arr_plot.shape[1])) 
    
    envMAW = np.repeat(envMAW_arr_plot, 10, axis=1)
    # envMAW_zoom = np.repeat(envMAW_zoom, 5, axis=0)
    
    DEM = np.repeat(DEM, 10, axis=1)
    # DEM = np.repeat(DEM, 5, axis=0)
    
    
    # CUT LASER OUTPUT TO FIT PHOTOS
    
    envMAW = envMAW[:,envMAW.shape[1]-1229:]
    DEM    = DEM[:,DEM.shape[1]-1229:]
    
    def rotate_matrix(matrix, angle):
        # Get the matrix shape
        rows, cols = matrix.shape[:2]
        
        # Calculate the rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
        
        # Rotate the matrix
        rotated_matrix = cv2.warpAffine(matrix, rotation_matrix, (cols, rows))
        
        return rotated_matrix
    
    envBAW = rotate_matrix(envBAW, -0.4)
    
    
    # SHIFT AND SCALE THE IMAGE
    # Define the transformation parameters
    scale = 0.20 # Enlargement scale
    dx = -30 # Shift in x direction
    dy = 6 # Shift in y direction
    
    # Create the transformation matrix
    M = np.float32([[scale, 0, dx], [0, scale, dy]])
    
    # Apply the transformation to img1 and store the result in img2
    rows, cols = envBAW.shape
    envBAW_sc = cv2.warpAffine(envBAW, M, (cols, rows))

    # Convert matrices to images
    # img1 = plt.imshow(channel, alpha=1.0)
    img2 = plt.imshow(envBAW_sc, cmap='Purples', alpha=0.5, vmin=0, vmax=1)
    img3 = plt.imshow(np.where(envMAW ==0, np.nan, envMAW), cmap='coolwarm', origin='upper', alpha=0.5)
    img3 = plt.imshow(np.where(np.isnan(DEM), 100, np.nan), cmap='bone', origin='upper', alpha=1.0)
    
    
    
    # Set title and show the plot
    plt.title(run_name)
    plt.axis('off')
    plt.savefig(os.path.join(report_dir, set_name, run_name + '_BAA_MAA.pdf'), dpi=600 )
    
    if index==1:
        plt.savefig(os.path.join(report_dir, set_name, set_name + '_report_BAA_MAA.pdf'), dpi=1200 )
        
    if nn==1:
        plt.savefig(os.path.join(report_dir, 'report_BAA_MAA.pdf'), dpi=1200 )
    
    plt.show()
    
    if index>1:
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