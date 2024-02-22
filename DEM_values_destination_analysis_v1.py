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
    
    output_dir = os.path.join(home_dir, 'output', run, 'DEM_analysis')
    if not(os.path.exists(output_dir)):
        os.mkdir(output_dir)
    plot_dir = os.path.join(output_dir, 'polts')
    if not(os.path.exists(plot_dir)):
        os.mkdir(plot_dir)
    
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
        hist_entire_dataset = hist_entire_dataset/np.nansum(hist_entire_dataset)
        
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
    
    # Plot the histogram with fixed upper and lower limits
    lower_limit_x, upper_limit_x = -40, 20  # Replace with desired x-axis limits
    lower_limit_y, upper_limit_y = 0, 0.1  # Replace with desired y-axis limits
    plt.xlim(lower_limit_x, upper_limit_x)
    plt.ylim(lower_limit_y, upper_limit_y)
    
    # Add labels and legend
    plt.xlabel('Data Points')
    plt.ylabel('Values')
    plt.title('Mean and Standard Deviation Plot - ' + run)
    plt.legend()
    
    # Save image and report
    plot_path = os.path.join(plot_dir, run + 'DEM_values_distribution.pdf')
    plt.savefig(plot_path, dpi=300)
    
    # Show the plot
    plt.show()
    