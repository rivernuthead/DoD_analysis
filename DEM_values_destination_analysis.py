#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 18:38:29 2024

@author: erri
"""

import os
import numpy as np
import matplotlib.pyplot as plt



# RUN NAMES
# runs = ['q07_1', 'q10_2','q10_3','q10_4', 'q15_2','q15_3', 'q20_2']

# runs = ['q10_2','q10_3','q10_4', 'q15_2','q15_3', 'q20_2']

# runs = ['q07_1', 'q10_2', 'q15_2', 'q20_2']

# runs = ['q07_1', 'q10_2', 'q15_2']

# runs = ['q20_2']

# runs = ['q15_3']

runs = ['q07_1']





home_dir = os.getcwd()


for run in runs:
    DEM_path = os.path.join(home_dir, 'surveys',run)
    stack_path = os.path.join(DEM_path, run + '_DEM_stack.npy')
    stack = np.load(stack_path)
    stack = np.where(stack==-999, np.nan, stack)
    dim_t, dim_y, dim_x = stack.shape
    
    bins = 80
    freq_dist_matrix = np.zeros((dim_t, bins))
    
    # DISTRIBUTION OF FREQUENCY
    for t in range(0,dim_t):
        matrix = stack[t,:,:]
        flat_matrix = matrix[~np.isnan(matrix)]
        
        # Compute normalized distributions for values greater than zero, lower than zero, and the entire dataset
        hist_greater_than_zero, edges_greater_than_zero = np.histogram(flat_matrix[flat_matrix >= 0], bins=40, density=True)
        hist_lower_than_zero, edges_lower_than_zero = np.histogram(flat_matrix[flat_matrix < 0], bins=40, density=True)
        hist_entire_dataset, edges_entire_dataset = np.histogram(flat_matrix, bins=80, density=True)
        
        freq_dist_matrix[t,:] = hist_entire_dataset
        
    mean_array = np.nanmean(freq_dist_matrix, axis=0)
    std_array = np.nanstd(freq_dist_matrix, axis=0)
    
    # Plot the mean as a line
    x_values = edges_entire_dataset[1:]+(edges_entire_dataset[1:]-edges_entire_dataset[:-1])/2
    plt.plot(x_values, mean_array, label='Mean', color='blue')
    
    # Fill the area around the mean with color representing the standard deviation
    plt.fill_between(x_values, mean_array - std_array, mean_array + std_array, color='lightblue', alpha=0.5, label='± 1 Std Dev')
    
    # Add a vertical black line at x=0
    plt.axvline(x=0, color='black', linestyle='--', label='x=0')

    # Add labels and legend
    plt.xlabel('Data Points')
    plt.ylabel('Values')
    plt.title('Mean and Standard Deviation Plot')
    plt.legend()
    
    # Show the plot
    plt.show()
    