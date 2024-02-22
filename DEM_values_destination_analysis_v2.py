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

runs = ['q07_1', 'q10_2', 'q15_3', 'q20_2']

# runs = ['q07_1', 'q10_2', 'q15_2']

# runs = ['q20_2']

# runs = ['q15_3']

# runs = ['q07_1']





home_dir = os.getcwd()


for run in runs:
    DEM_path = os.path.join(home_dir, 'surveys',run)
    stack_path = os.path.join(DEM_path, run + '_DEM_stack.npy')
    stack = np.load(stack_path)
    stack = np.where(stack==-999, np.nan, stack)
    dim_t, dim_y, dim_x = stack.shape
    
    
    flat_matrix = stack[~np.isnan(stack)]
        
    # Compute normalized distributions for values greater than zero, lower than zero, and the entire dataset
    hist_greater_than_zero, edges_greater_than_zero = np.histogram(flat_matrix[flat_matrix >= 0], bins=40, density=True)
    hist_lower_than_zero, edges_lower_than_zero = np.histogram(flat_matrix[flat_matrix < 0], bins=40, density=True)
    hist_entire_dataset, edges_entire_dataset = np.histogram(flat_matrix, bins=80, density=True)
    
    # Plot the mean as a line

    # Plot the histogram with fixed upper and lower limits
    lower_limit_x, upper_limit_x = -40, 20  # Replace with desired x-axis limits
    lower_limit_y, upper_limit_y = 0, 0.5  # Replace with desired y-axis limits
    
    plt.hist(flat_matrix, bins=80, density=True, color='blue', edgecolor='black', range=(lower_limit_x, upper_limit_x))
    plt.xlim(lower_limit_x, upper_limit_x)
    plt.ylim(lower_limit_y, upper_limit_y)
    # Add a vertical black line at x=0
    plt.axvline(x=0, color='black', linestyle='--', label='x=0')

    # Add labels and legend
    plt.xlabel('Data Points')
    plt.ylabel('Values')
    plt.title('Mean and Standard Deviation Plot')
    plt.legend()
    
    # Show the plot
    plt.show()
    