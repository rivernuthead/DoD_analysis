#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 16:07:36 2024

@author: erri
"""
import os
import numpy as np


runs = ['q07_1', 'q10_2','q10_3','q10_4', 'q15_2','q15_3', 'q20_2']

# Run parameter
NaN = -999
for run in runs:
    
    # Step 1: List all files in the folder
    home_dir = os.getcwd()
    folder_path = os.path.join(home_dir, 'surveys',run)
    files = os.listdir(folder_path)
    
    # Step 2: Filter files based on a common part in the name
    common_part = "matrix_bed_norm"
    filtered_files = [file for file in files if common_part in file and file.endswith('.txt')]
    
    list1 = []
    list2 = []
    for name in filtered_files:
    
        if len(name) == len(filtered_files[0]):
            list1 = np.append(list1, name)
        else:
            list2 = np.append(list2, name)
    filtered_files_sorted = list(list1) + list(list2)
    
    # Step 4: Define and applly mask
    array_dim_x = []
    array_dim_y = []
    for f in filtered_files_sorted:
        path_DEM = os.path.join(folder_path, f)
        DEM = np.loadtxt(path_DEM,
                          # delimiter=',',
                          skiprows=8
                          )
        array_dim_x = np.append(array_dim_x, DEM.shape[0])
        array_dim_y = np.append(array_dim_y, DEM.shape[1])

    # Define target dimension:
    shp_target_x, shp_target_y = int(min(array_dim_x)), int(min(array_dim_y))

    arr_shape = np.array([shp_target_x, shp_target_y]) # Define target shape

    # array mask for filtering data outside the channel domain
    # Different mask will be applied depending on the run due to different ScanArea
    # used during the laser surveys
    runs_list = ['q10_1', 'q10_2', 'q15_1', 'q20_1', 'q20_2'] # Old runs with old ScanArea
    array_mask_name, array_mask_path = 'array_mask.txt', home_dir # Mask for runs 07 onwards

    if run in runs_list:
        array_mask_name, array_mask_path = 'array_mask_0.txt', home_dir

    # Load mask
    array_mask = np.loadtxt(os.path.join(array_mask_path, array_mask_name))
    # Reshape mask:
    array_mask_rshp = array_mask[:shp_target_x,:shp_target_y] # Array mask reshaped

    # Create array mask:
    # - array_mask: np.array with 0 and 1
    # - array_mask_nan: np.array with np.nan and 1
    array_mask_rshp = np.where(array_mask_rshp==NaN, 0, 1) # Convert in mask with 0 and 1
    array_mask_rshp_nan = np.where(array_mask_rshp==0, np.nan, 1) # Convert in mask with np.nan and 1
    
    # Step 5: Load each text file as a numpy matrix and store in a list
    matrix_list = []
    for file in filtered_files:
        file_path = os.path.join(folder_path, file)
        
        # Assuming the data in the text file is organized as rows and columns
        # Modify the loading logic based on your actual file format
        matrix = np.loadtxt(file_path, skiprows=8)
        matrix = matrix*array_mask_rshp_nan
        
        matrix_list.append(matrix)
    
    # Step 5: Convert the list of matrices into a 3D numpy array (stack)
    stacked_matrix = np.stack(matrix_list, axis=0)
    
    # Step 6: Save the stacked matrix as a Numpy binary file
    output_file_path = os.path.join(folder_path, run + '_DEM_stack.npy')
    np.save(output_file_path, stacked_matrix)
