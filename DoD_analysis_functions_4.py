#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 11:16:17 2022

@author: erri
"""

import numpy as np
from skimage import morphology
import cv2


def remove_small_objects(matrix, object_size, connectivity):
    '''
    

    Parameters
    ----------
    matrix : TYPE
        DESCRIPTION.
    object_size : TYPE
        DESCRIPTION.
    connectivity : TYPE
        DESCRIPTION.

    Returns
    -------
    matrix_out : TYPE
        DESCRIPTION.

    '''
    # Convert matrix in bool matrix
    matrix_bool = np.where(matrix>0, 1, 0)
    matrix_bool = np.where(matrix<0, 1, matrix_bool)
    matrix_bool = np.array(matrix_bool, dtype='bool') # Convert in array of bool
    
    matrix_mask = morphology.remove_small_objects(matrix_bool, object_size, connectivity) # Morphological analysis
    
    matrix_out = matrix_mask*matrix
    
    matrix_out_gis = np.where(np.isnan(matrix_out), -999, matrix_out)
    
    return matrix_out

def fill_small_holes(matrix, avg_target_kernel, area_threshold, connectivity, filt_threshold):
    
    # Convert matrix in bool matrix
    matrix_bool = np.where(matrix>0, 1, 0)
    matrix_bool = np.where(matrix<0, 1, matrix_bool)
    matrix_bool = np.array(matrix_bool, dtype='bool') # Convert in array of bool
    
    # Set target and average
    ker=np.ones((avg_target_kernel,avg_target_kernel), np.float32)/(avg_target_kernel**2)
    matrix_target = np.where(np.isnan(matrix), 0, matrix)
    
    matrix_target = cv2.filter2D(src=matrix_target,ddepth=-1, kernel=ker)

    # Perform morphological analysis
    matrix_out = morphology.remove_small_holes(matrix_bool, area_threshold=area_threshold, connectivity=connectivity)
    
    # Apply the target where holes have to be filled
    matrix_out = np.where(matrix_bool!=matrix_out, matrix_target, matrix)
    
    # matrix_out = matrix_target*matrix_out
    
    return matrix_out, matrix_target
    


def nature_checker(matrix, threshold, round_value, NaN):
    '''
    
    Parameters
    ----------
    matrix : 2D numpy matrix
        DESCRIPTION.
    round_value : integer
        The number of decimal digit at which round the output values
    NaN : real
        The NaN value to use in the gis readable conversion of the output matrix

    Returns
    -------
    matrix : 2D numpy matrix
        DESCRIPTION.
    matrix_out_gis : 2D numpy matrix
        GIS readable matrix where np.nans where replaced with NaN value
        (To be completed, the header must be added)
        
    This function check the nature of the analyzed cell: if there are a number
    of cells with the same nature of the analyzed cell grater than the threshold
    the function confirm the value, otherwise the function will zero the cell value
    '''
    dim_y, dim_x = matrix.shape # Extract matrix dimensions
    
    matrix_out = np.copy(matrix) # Initialize output matrix as a copy of the input matrix
    
    # Create the analysis domain, padding value as the matrix edge:
    matrix_dom = np.pad(matrix, 1, mode='edge') # Create neighbourhood analysis domain
    
    # Cycle over all the matrix cells
    for i in range(0,dim_y):
        for j in range(0,dim_x):
            if matrix[i,j]!=0 and not(np.isnan(matrix[i,j])): # If the analyzed cell value is neither zero nor np.nan
                ker = np.array([[matrix_dom[i, j], matrix_dom[i, j + 1], matrix_dom[i, j + 2]],
                                [matrix_dom[i + 1, j], matrix_dom[i + 1, j + 1], matrix_dom[i + 1, j + 2]],
                                [matrix_dom[i + 2, j], matrix_dom[i + 2, j + 1], matrix_dom[i + 2, j + 2]]])
                if not((matrix[i,j] > 0 and np.count_nonzero(ker > 0) >= threshold) or (matrix[i,j] < 0 and np.count_nonzero(ker < 0) >= threshold)): # if not(the nature is confirmed)
                    # So if the nature of the selected cell is not confirmed...
                    matrix_out[i,j] = 0
    
    # Round the output matrix values
    matrix_out = np.round(matrix_out, 1)
    
    # Create a GIS readable DoD mean (np.nan as -999)
    matrix_out_gis = np.where(np.isnan(matrix_out), NaN, matrix_out)
    
    return matrix_out, matrix_out_gis

def repeat_columns(matrix, N):
    '''

    Parameters
    ----------
    matrix : TYPE
        DESCRIPTION.
    N : TYPE
        DESCRIPTION.

    Returns
    -------
    new_matrix : TYPE
        DESCRIPTION.

    '''
    new_matrix = np.zeros((matrix.shape[0], matrix.shape[1]*N))
    for i in range(matrix.shape[1]):
        new_matrix[:, i*N:i*N+N] = matrix[:, i:i+1]
    return new_matrix

def test(matrix):
    '''
    This function remove isolate pixels whene surrounded by zeros or by active
    pixels with opposite nature

    Parameters
    ----------
    matrix : TYPE
        DESCRIPTION.

    Returns
    -------
    matrix_out : TYPE
        DESCRIPTION.

    '''
    dim_y, dim_x = matrix.shape # Extract matrix dimensions
    
    matrix_out = np.copy(matrix) # Initialize output matrix as a copy of the input matrix
    
    # Create the analysis domain, padding value as the matrix edge:
    matrix_dom = np.pad(matrix, 1, mode='edge') # Create neighbourhood analysis domain
    
    # Cycle over all the matrix cells
    for i in range(0,dim_y):
        for j in range(0,dim_x):
            if matrix[i,j]!=0 and not(np.isnan(matrix[i,j])): # If the analyzed cell value is neither zero nor np.nan
                ker = np.array([[matrix_dom[i, j], matrix_dom[i, j + 1], matrix_dom[i, j + 2]],
                                [matrix_dom[i + 1, j], matrix_dom[i + 1, j + 1], matrix_dom[i + 1, j + 2]],
                                [matrix_dom[i + 2, j], matrix_dom[i + 2, j + 1], matrix_dom[i + 2, j + 2]]])
                
                if matrix[i,j]>0 and np.count_nonzero(ker > 0)==1: # If the core cell is positive and is the only one
                    matrix_out[i,j]=0 # Set the core cell as zero
                elif matrix[i,j]<0 and np.count_nonzero(ker < 0)==1: # If the core cell is negative and is the only one
                    matrix_out[i,j]=0 # Set the core cell as zero
                    
    return matrix_out
    

def test2(matrix):
    '''
    This function remove peninsular pixel along the x axis
    First it check if the upper and lower cells are zero
    
      0      0      0
      0    -2.6   -3.1
      0      0    -2.2
      
    If these cell are zero the function checks if the cells
    with the same nature of the target cell are more than 4.
    In not, the target cell is set as 0.
    
    Parameters
    ----------
    matrix : TYPE
        DESCRIPTION.

    Returns
    -------
    matrix_out : TYPE
        DESCRIPTION.

    '''
    dim_y, dim_x = matrix.shape # Extract matrix dimensions
    
    matrix_out = np.copy(matrix) # Initialize output matrix as a copy of the input matrix
    
    # Create the analysis domain, padding value as the matrix edge:
    matrix_dom = np.pad(matrix, 1, mode='edge') # Create neighbourhood analysis domain
    matrix_dom = np.where(np.isnan(matrix_dom), 0, matrix_dom)
    
    # Cycle over all the matrix cells
    for i in range(0,dim_y):
        for j in range(0,dim_x):
            if matrix[i,j]!=0 and not(np.isnan(matrix[i,j])): # If the analyzed cell value is neither zero nor np.nan
                ker = np.array([[matrix_dom[i, j], matrix_dom[i, j + 1], matrix_dom[i, j + 2]],
                                [matrix_dom[i + 1, j], matrix_dom[i + 1, j + 1], matrix_dom[i + 1, j + 2]],
                                [matrix_dom[i + 2, j], matrix_dom[i + 2, j + 1], matrix_dom[i + 2, j + 2]]])
                
                if matrix[i,j]>0 and matrix_dom[i, j + 1]==0 and matrix_dom[i + 2, j + 1]==0 and np.count_nonzero(ker>0)<5: # If the core cell is positive and there is at least another positive cell
                    matrix_out[i,j]=0 # Set the core cell as zero
                elif matrix[i,j]<0 and matrix_dom[i, j + 1]==0 and matrix_dom[i + 2, j + 1]==0 and np.count_nonzero(ker<0)<5: # If the core cell is negative and there is at least another positive cell
                    matrix_out[i,j]=0 # Set the core cell as zero
    
    return matrix_out


def rescaling_plot(matrix):
    matrix_out = np.repeat(matrix, 10, axis=1)
    matrix_out = np.where(matrix_out==0, np.nan, matrix_out)
    return matrix_out