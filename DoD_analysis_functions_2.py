#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 11:16:17 2022

@author: erri
"""

import numpy as np


def spatial_weighted_average(matrix, round_value, NaN):
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
    matrix_out : 2D numpy matrix
        DESCRIPTION
    matrix_out_gis : 2D numpy matrix
        GIS readable matrix where np.nans where replaced with NaN value
        (To be completed, the header must be added)
        
    
    This function perform a weighted spatial average filter on a given 2D matrix.
    The w matrix below define the weights
    '''
    
    dim_y, dim_x = matrix.shape # Extract matrix dimensions
    
    matrix_out = np.zeros(matrix.shape) # Initialize output matrix 
    
    # Create the spatial average domain, padding value as the matrix edge:
    matrix_dom = np.pad(matrix, 1, mode='edge') # Create neighbourhood analysis domain
    
    # Cycle over all the matrix cells
    for i in range (0, dim_y):
        for j in range (0, dim_x):
            
            if np.isnan(matrix[i, j]): # If input data cell is np.nan keep it as np.nan
                matrix_out[i, j] = np.nan
                
            else:
                # Define analysis kernel
                ker = np.array([[matrix_dom[i, j], matrix_dom[i, j + 1], matrix_dom[i, j + 2]],
                              [matrix_dom[i + 1, j], matrix_dom[i + 1, j + 1], matrix_dom[i + 1, j + 2]],
                              [matrix_dom[i + 2, j], matrix_dom[i + 2, j + 1], matrix_dom[i + 2, j + 2]]])
                # Define weighted average parameters
                w = np.array([[0, 1, 0],
                              [0, 2, 0],
                              [0, 1, 0]])
                w_norm = w / (sum(sum(w)))  # Normalizing weight matrix
                
                # Fill output matrix with the weighted averaged data
                matrix_out[i, j] = np.nansum(ker*w_norm)
    
    # Round the output matrix values
    matrix_out = np.round(matrix_out, 1)
    
    # Create a GIS readable DoD mean (np.nan as -999)
    matrix_out_gis = np.where(np.isnan(matrix_out), NaN, matrix_out)
    
    return matrix_out, matrix_out_gis


def isolated_killer(matrix, threshold, round_value, NaN):
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
        
    This function will zeroing all the value surrounded by more than 7 zero cells

    '''
    dim_y, dim_x = matrix.shape # Extract matrix dimensions
    
    matrix_out = np.copy(matrix) # Initialize output matrix as a copy of the input matrix
    
    # Create the analysis domain, padding value as the matrix edge:
    matrix_dom = np.pad(matrix, 1, mode='constant', constant_values=0) # Create neighbourhood analysis domain
    
    # Cycle over all the matrix cells
    for i in range (0, dim_y):
        for j in range (0, dim_x):
            if matrix[i,j] != 0 and not(np.isnan(matrix[i,j])): # Limiting the analysis to non-zero numbers and non-np.nan
                # Create kernel
                ker = np.array([[matrix_dom[i, j], matrix_dom[i, j + 1], matrix_dom[i, j + 2]],
                                [matrix_dom[i + 1, j], matrix_dom[i + 1, j + 1], matrix_dom[i + 1, j + 2]],
                                [matrix_dom[i + 2, j], matrix_dom[i + 2, j + 1], matrix_dom[i + 2, j + 2]]])
                zero_count = np.count_nonzero(ker == 0) + np.count_nonzero(np.isnan(ker))
                if zero_count >= threshold:
                    matrix_out[i,j] = 0
                else:
                    pass
    
    # Round the output matrix values
    matrix_out = np.round(matrix_out, 1)
    
    # Create a GIS readable DoD mean (np.nan as -999)
    matrix_out_gis = np.where(np.isnan(matrix_out), NaN, matrix_out)
    
    return matrix_out, matrix_out_gis


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

def isolated_filler(matrix, threshold, round_value, NaN):
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
        
    This function fill all the zero value cell surrounded by no-zero value cells
    following this procedure:
        if the analyzed call is zero and in the kernel do not appear any np.nan
        value the function count the number of positive and negative values in
        the kernel. If there are a number of positive or negative values grater
        than the threshold the analyzed cell will be filled with the mean of
        the positive or negative values in the kernel according with which of
        them are grater, in number, than the threshold.

    '''
    dim_y, dim_x = matrix.shape # Extract matrix dimensions
    
    matrix_out = np.copy(matrix) # Initialize output matrix as a copy of the input matrix
    
    # Create the analysis domain, padding value as the matrix edge:
    matrix_dom = np.pad(matrix, 1, mode='edge') # Create neighbourhood analysis domain
    
    # Cycle over all the matrix cells
    for i in range (0, dim_y):
        for j in range (0, dim_x):
            if matrix[i,j] == 0: # If the analyzed cell is zero
                # Create kernel
                ker = np.array([[matrix_dom[i, j], matrix_dom[i, j + 1], matrix_dom[i, j + 2]],
                                [matrix_dom[i + 1, j], matrix_dom[i + 1, j + 1], matrix_dom[i + 1, j + 2]],
                                [matrix_dom[i + 2, j], matrix_dom[i + 2, j + 1], matrix_dom[i + 2, j + 2]]])
                if np.isnan(ker).any(): # If there are np.nan value in the kernel
                    pass # Nothing will change
                # So if there are not np.nan...
                elif np.count_nonzero(ker > 0) >= threshold: # ... and the number of positive values is grather than the threshold:
                    matrix_out[i,j] = np.mean((ker>0)*ker) # Fill the zero value analyzed cell with the mean of the positive values
                elif np.count_nonzero(ker < 0) >= threshold: # Otherwise if the number of negative values is grather than the threshold:
                    matrix_out[i,j] = np.mean((ker<0)*ker) # Fill the zero value analyzed cell with the mean of the negative values
                else:
                    pass
    
    # Round the output matrix values
    matrix_out = np.round(matrix_out, 1)
    
    # Create a GIS readable DoD mean (np.nan as -999)
    matrix_out_gis = np.where(np.isnan(matrix_out), NaN, matrix_out)
                    
    return matrix_out, matrix_out_gis



def island_destroyer(matrix, window_dim, round_value, NaN):
    '''
    
    Parameters
    ----------
    matrix : 2D numpy matrix
        DESCRIPTION.
    window_dim : integer
        kernel window dimension
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
        
    This function will zeroing all the islands that are detected as surrounded
    by zeros given the kernel analysis dimension

    '''
    dim_y, dim_x = matrix.shape # Extract matrix dimensions
    
    matrix_out = np.copy(matrix) # Initialize output matrix as a copy of the input matrix
    
    # Create the analysis domain, converting np.nan to zero:
    matrix_dom = np.where(np.isnan(matrix), 0, matrix)
    
    # Cycle over all the matrix cells
    for i in range (0, dim_y-window_dim):
        for j in range (0, dim_x-window_dim):
            # Create kernel
            ker = matrix_dom[i:i+window_dim,j:j+window_dim] # Define the moving kernel with a window_dim x window_dim dimension
            ker_edge = np.hstack((ker[0,:], ker[window_dim-1,:], ker[1:-1,0], ker[1:-1,window_dim-1]))
            zero_edge_count = np.count_nonzero(ker_edge==0)
            if zero_edge_count == 4*(window_dim-1):
                matrix_out[i:i+window_dim,j:j+window_dim] = 0
            else:
                pass

    # Round the output matrix values
    matrix_out = np.round(matrix_out, 1)
    
    # Create a GIS readable DoD mean (np.nan as -999)
    matrix_out_gis = np.where(np.isnan(matrix_out), NaN, matrix_out)
    
    return matrix_out, matrix_out_gis



ker = np.array([[0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 3, 1, 3, 0],
                [0, -2, 1, 4, 1, 1, 0],
                [0, 2, 4 ,0, 0, 1, 0],
                [0, 4, 5, 6, 1, 7, 0],
                [0, 2, 4, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0]])

output = island_destroyer(ker, 5, 1, -999)