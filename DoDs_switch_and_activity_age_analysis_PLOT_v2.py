#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 17:02:28 2023

@author: erri

This script analyze the DoD at a give delta timespan and compute:
    1. The number of switch that occours during the run and provieds a map
    of the spatially distributed number of switch
    2. Compute the frequency distribution of the number of switch

NB: This script do not includes zero in the switch calculation:
    so [1,0,1] is not considered as a switch between 1 and 0.
    

"""

# IMPORT PACKAGES
import os
import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.stats import poisson
from scipy.optimize import curve_fit
from period_function_v3 import *


# VERY FIRST SETUP
start = time.time() # Set initial time

plot_mode=[
    'periods_dist',
    # 'switch_number_dist',
    'poisson_dist'
    ]

# SINGLE RUN NAME
runs = ['q07_1']
# runs = ['q10_2']
# runs = ['q10_3']
# runs = ['q10_4']
# runs = ['q15_2']
# runs = ['q15_3']
# runs = ['q20_2']
runs = ['q07_1', 'q10_2', 'q15_3', 'q20_2']
runs = ['q10_3', 'q10_2', 'q10_4']  
# runs = ['q07_1', 'q10_2', 'q10_3', 'q10_4', 'q15_2', 'q15_3', 'q20_2']

# DEFINE DoD TIMESPAN
t_span = 0

'''
INPUT:
    the script takes in input a stack where all the DoDs are stored in a structure as shown below:
        
    DoD input stack structure:
        
        DoD_stack[time,y,x,delta]
        DoD_stack_bool[time,y,x,delta]
        
         - - - 0 - - - - 1 - - - - 2 - - - - 3 - - - - 4 - - - - 5 - - - - 6 - - - - 7 - - - - 8 - -  >    delta
      0  |  DoD 1-0   DoD 2-0   DoD 3-0   DoD 4-0   DoD 5-0   DoD 6-0   DoD 7-0   DoD 8-0   DoD 9-0
      1  |  DoD 2-1   DoD 3-1   DoD 4-1   DoD 5-1   DoD 6-1   DoD 7-1   DoD 8-1   DoD 9-1
      2  |  DoD 3-2   DoD 4-2   DoD 5-2   DoD 6-2   DoD 7-2   DoD 8-2   DoD 9-2
      3  |  DoD 4-3   DoD 5-3   DoD 6-3   DoD 7-3   DoD 8-3   DoD 9-3
      4  |  DoD 5-4   DoD 6-4   DoD 7-4   DoD 8-4   DoD 9-4
      5  |  DoD 6-5   DoD 7-5   DoD 8-5   DoD 9-5
      6  |  DoD 7-6   DoD 8-6   DoD 9-6
      7  |  DoD 8-7   DoD 9-7
      8  |  DoD 9-8
         |
         v
        
         time
        
'''
# INITIALIZE ARRAY
activated_px   = []
deactivated_px = []
      
for run in runs:
    print()
    print()
    print(run, ' is running...')
    # FOLDER SETUP
    home_dir = os.getcwd() # Home directory
    report_dir = os.path.join(home_dir, 'output')
    DoDs_folder = os.path.join(home_dir,'output', 'DoDs', 'DoDs_stack') # Input folder
    output_dir = os.path.join(report_dir, 'report_'+run, 'switch_analysis')
    plot_dir = os.path.join(output_dir, 'plot')
    if not(os.path.exists(output_dir)):
        os.mkdir(output_dir)
    if not(os.path.exists(plot_dir)):
        os.mkdir(plot_dir)
        
    # INITIALIZE ARRAYS

    
    ###########################################################################

    # IMPORT DoD STACK AND DoD BOOL STACK
    DoDs_folder = os.path.join(home_dir,'output', 'DoDs', 'DoDs_stack') # Input folder
    stack_name = 'DoD_stack' + '_' + run + '.npy' # Define stack name
    stack_bool_name = 'DoD_stack' + '_bool_' + run + '.npy' # Define stack bool name
    stack_path = os.path.join(DoDs_folder,stack_name) # Define stack path
    stack_bool_path = os.path.join(DoDs_folder,stack_bool_name) # Define stack bool path
    
    stack = np.load(stack_path) # Load DoDs stack
    stack_bool = np.load(stack_bool_path) # Load DoDs boolean stack
    
    # Load diff stack for new activated or deactivated area
    # 1=activated, -1=deactivated, 0=no_changes
    diff_stack = np.load(os.path.join(home_dir, 'output','report_'+run, run + '_diff_stack.npy'))
    diff_stack = diff_stack[:,:,:,0]

    
    dim_t, dim_y, dim_x, dim_d = stack.shape # Define time dimension, crosswise dimension and longitudinal dimension
    
    # COMPUTE THE ACTIVE AREA
    act_stack = np.abs(stack_bool[:,:,:,0])
    activity_arr   = np.nansum(act_stack, axis=2)
    activity_arr   = np.nansum(activity_arr, axis=1)
    mean_activity_area = np.nanmean(activity_arr)
    
    # COMPUTE AND STORE THE ACTIVATED AND DEACTIVATED NUMBER OF PIXEL
    diff_stack_act = diff_stack*(diff_stack>0) # Extract activated pixel only
    diff_stack_deact = diff_stack*(diff_stack<0) # Extract deactivated pixel only
    
    act_arr = np.nansum(diff_stack_act, axis=2)
    act_arr = np.nansum(act_arr, axis=1)
    act_arr_rel = act_arr/activity_arr[:-1]
    act_arr_abs = act_arr/dim_x/120
    
    deact_arr = np.nansum(diff_stack_deact, axis=2)
    deact_arr = np.nansum(deact_arr, axis=1)
    deact_arr_rel = deact_arr/activity_arr[:-1]
    deact_arr_abs = deact_arr/dim_x/120
    
    # Save report as txt file
    np.savetxt(os.path.join(output_dir, run+'_relative_activated_pixels.txt'), act_arr_rel, fmt='%.4f')
    np.savetxt(os.path.join(output_dir, run+'_relative_deactivated_pixel.txt'), deact_arr_rel, fmt='%.4f')
    np.savetxt(os.path.join(output_dir, run+'_absolute_activated_pixels.txt'), act_arr_abs, fmt='%.4f')
    np.savetxt(os.path.join(output_dir, run+'_absolute_deactivated_pixel.txt'), deact_arr_abs, fmt='%.4f')
    
    # IMPORT PIXEL AGE ANALYSIS OUTPUTS
    
    '''
    n_0: int - Number of zeros before the first non-zero value
    time_array: np.array - Length with sign of the active periods
    consecutive_minus_ones_array: np.array - Length of consecutive -1 values
    consecutive_zeros_array: np.array - Length of consecutive 0 values
    consecutive_ones_array: np.array - Length of consecutive +1 values
    distances_array: np.array - Distance between switches
    switch_counter: int - Number of switches
    '''
    
    n_0_matrix = np.load(os.path.join(output_dir, run + '_tspan_' + str(t_span) + '_n_zeros.npy'))
    time_stack = np.load(os.path.join(output_dir, run + '_tspan_' + str(t_span) + '_time_stack.npy'))
    consecutive_minus_ones_stack = np.load(os.path.join(output_dir, run + '_tspan_' + str(t_span) + '_consecutive_minus_ones_stack.npy'))
    consecutive_zeros_stack = np.load(os.path.join(output_dir, run + '_tspan_' + str(t_span) + '_consecutive_zeros_stack.npy'))
    consecutive_ones_stack = np.load(os.path.join(output_dir, run + '_tspan_' + str(t_span) + '_consecutive_ones_stack.npy'))
    distances_stack = np.load(os.path.join(output_dir, run + '_tspan_' + str(t_span) + '_distances_stack.npy'))
    switch_matrix = np.load(os.path.join(output_dir, run + '_tspan_' + str(t_span) + '_switch_matrix.npy'))

    
    ###########################################################################
    # PLOTS
    ###########################################################################
    
    if 'periods_dist' in plot_mode:
        
        # COMPUTE SCOUR AND DEPOSITION MEDIAN
        time_array_cld    = time_stack[(time_stack != 0) & (~np.isnan(time_stack))] # Trim np.nan and zeros
        sco_time_array    = time_array_cld*(time_array_cld<0)
        sco_time_array    = sco_time_array[sco_time_array != 0]
        dep_time_array    = time_array_cld*(time_array_cld>0)
        dep_time_array    = dep_time_array[dep_time_array != 0]
        sco_median        = np.median(sco_time_array)
        dep_median        = np.median(dep_time_array)
        sco_mean          = np.mean(sco_time_array)
        dep_mean          = np.mean(dep_time_array)
        sco_percentile_25 = np.percentile(sco_time_array,25)
        dep_percentile_75 = np.percentile(dep_time_array,75)
        sco_percentile_10 = np.percentile(sco_time_array,10)
        dep_percentile_90 = np.percentile(dep_time_array,90)
        
        # PRINT SCOUR AND DEPOSITION MEDIAN
        print('Periods length')
        print('Scour median: ', sco_median)
        print('Deposition median: ', dep_median)
        print('Scour mean: ', sco_mean)
        print('Deposition mean: ', dep_mean)
        print('Scour 25° percentile: ', sco_percentile_25)
        print('Deposition 75° percentile: ', dep_percentile_75)
        print('Scour 10° percentile: ', sco_percentile_10)
        print('Deposition 90° percentile: ', dep_percentile_90)
        
        report = np.array([sco_median, dep_median, sco_percentile_25, dep_percentile_75, sco_percentile_10, dep_percentile_90])
        print('sco_median, dep_median, sco_percentile_25, dep_percentile_75, sco_percentile_10, dep_percentile_90')
        print(report)
        
        # COMPUTE THE DISTRIBUTION
        num_bins_overall = 21
        # bin_edges = [-60.0,-8.6,-5.8,-4.0,-2.6,-1.3,1.3,2.3,3.7,5.5,8.3,60.0]
        hist_range = (-10, 10)
        x_values = np.linspace(-10, 10, num_bins_overall)
        
        hist_array = np.copy(time_stack)
        # hist_array = hist_array[~np.isnan(hist_array)] # Trim np.nan
        hist_array = hist_array[(hist_array != 0) & (~np.isnan(hist_array))] # Trim np.nan and zeros
        hist, bin_edges = np.histogram(hist_array, bins=num_bins_overall, range=hist_range)
        # dist_time_matrix = hist/mean_activity_area # Divide the number of switch by the mean active area
        dist_time_matrix = hist/np.nansum(hist)
        
        # Define colors based on x-axis values
        colors = np.where(x_values < 0, 'red', 'blue')
        
        plt.bar(x_values, dist_time_matrix, label='period duration', color=colors, linewidth=0.6)
        
        # Insert medians line
        plt.axvline(x=sco_median, color='red', linestyle='--')
        plt.axvline(x=dep_median, color='blue', linestyle='--')
        
        # Inser text
        plt.text(sco_median - 5, max(dist_time_matrix) + 0.01, 'median: ' + str(sco_median), color='red')
        plt.text(dep_median + 0.5, max(dist_time_matrix) + 0.01, 'median: ' + str(dep_median), color='blue')
        
        # Add a chart title
        plt.title(run + ' - time span: ' + str(t_span))
        
        # Set the y axis limits
        lower_limit = 0.0
        upper_limit = 0.3
        plt.ylim(lower_limit,upper_limit)
        
        # Set the y-axis ticks at intervals of 0.5
        plt.xticks(np.linspace(-10, 10 ,21))
        
        # Add labels and legend
        plt.xlabel('Time span between switches')
        plt.ylabel('Y Values')
        # plt.legend(fontsize=8)
        
        # Save the figure to a file (e.g., 'scatter_plot.png')
        plt.savefig(os.path.join(plot_dir, run + '_tspan' + str(t_span) + '_activity_periods_distribution.pdf'), dpi=300)
        # plt.savefig(os.path.join(report_dir, 'overall_distribution' ,run + '_'+str(i)+ '_overall_dist_chart.pdf'), dpi=300)
        plt.show()
        
        print()
        
        
        
        # POISSON DISTRIBUTION FITTING - SCOUR AND DEPOSITION
        
        data_sco = np.copy(abs(sco_time_array))
        data_dep = np.copy(dep_time_array)
        
        # HIST
        sco_hist_array = data_sco[(data_sco != 0) & (~np.isnan(data_sco))] # Trim np.nan and zeros
        dep_hist_array = data_dep[(data_dep != 0) & (~np.isnan(data_dep))] # Trim np.nan and zeros
        
        sco_hist, bin_edges = np.histogram(sco_hist_array, bins=dim_t, range=hist_range)
        dep_hist, bin_edges = np.histogram(dep_hist_array, bins=dim_t, range=hist_range)
        
        sco_hist = sco_hist/np.nansum(sco_hist)
        dep_hist = dep_hist/np.nansum(dep_hist)
        
        # POISSON DIST
        if 'poisson_dist' in plot_mode:
            
            # the bins have to be kept as a positive integer because poisson is a positive integer distribution
            bins = np.arange(10) + 0.5
            
            
            # plot poisson-deviation with fitted parameter
            x_plot = np.arange(1, 10)
            
            
            
            # Create subplots with 1 row and 2 columns
            fig, axs = plt.subplots(1, 2, figsize=(10, 4))
            
            
            sco_entries, bin_edges, patches = axs[0].hist(data_sco, bins=bins, density=True, label="Scour", width=0.8, color='red')
            dep_entries, bin_edges, patches = axs[1].hist(data_dep, bins=bins, density=True, label="Fill", width=0.8, color='blue')
            
            # calculate bin centers
            middles_bins = (bin_edges[1:] + bin_edges[:-1]) * 0.5
            
            
            def fit_function(k, lamb):
                # The parameter lamb will be used as the fit parameter
                return poisson.pmf(k, lamb)
            
            
            # fit with curve_fit
            sco_parameters, sco_cov_matrix = curve_fit(fit_function, middles_bins, sco_entries)
            dep_parameters, dep_cov_matrix = curve_fit(fit_function, middles_bins, dep_entries)
            
            
            
            
            # # Plot the first histogram
            # axs[0].bar(x_plot, sco_hist)
            
            axs[0].plot(
                    x_plot,
                    fit_function(x_plot, *sco_parameters),
                    marker="D",
                    linestyle="-",
                    color="green",
                    label="Fit result",
                    )
            
            axs[1].plot(
                    x_plot,
                    fit_function(x_plot, *dep_parameters),
                    marker="D",
                    linestyle="-",
                    color="green",
                    label="Fit result",
                    )
            
            # Add text in a box in the upper right corner
            sco_text_content = 'Poisson parameter: ' + str(np.round(sco_parameters,3))
            dep_text_content = 'Poisson parameter: ' + str(np.round(dep_parameters,3))
            
            box_properties = dict(boxstyle='round', facecolor='white', edgecolor='black', alpha=0.7)
            
            axs[0].annotate(sco_text_content, xy=(0.95, 0.75), xycoords='axes fraction',
                          fontsize=10, ha='right', va='top', bbox=box_properties)
            axs[1].annotate(dep_text_content, xy=(0.95, 0.75), xycoords='axes fraction',
                          fontsize=10, ha='right', va='top', bbox=box_properties)
            
            
            # Modify the x-axis tick distance in both subplots
            tick_distance = 1
            axs[0].set_xticks(x_plot[::tick_distance])
            axs[0].set_xticklabels(x_plot[::tick_distance])
            axs[1].set_xticks(x_plot[::tick_distance])
            axs[1].set_xticklabels(x_plot[::tick_distance])
            
            plt.suptitle(run + ' _ Switch number - Poisson distribution fit')
            
            
            axs[0].legend()
            axs[1].legend()
            
            # Save the figure to a file (e.g., 'scatter_plot.png')
            plt.savefig(os.path.join(plot_dir, run + '_tspan' + str(t_span) + '_sco_dep_distribution_poisson.pdf'), dpi=300)
            
            plt.show()
            
            
            
        
        
    if 'switch_number_dist' in plot_mode:
        
        # COMPUTE SWITCH NUMBER DISTRIBUTION
        
        
        # COMPUTE THE DISTRIBUTION
        num_bins_overall = 10
        hist_range = (1, 10)
        x_values = np.linspace(1, 10, num_bins_overall)
        
        hist_array = np.copy(switch_matrix)
        hist_array = hist_array[(hist_array != 0) & (~np.isnan(hist_array))] # Trim np.nan and zeros
        median = np.median(hist_array)
        hist, bin_edges = np.histogram(hist_array, bins=num_bins_overall, range=hist_range)
        # dist_switch_matrix = hist/mean_activity_area # Divide the number of switch by the mean active area
        dist_switch_matrix = hist/np.nansum(hist)
        
        # PLOT THE DISTRIBUTION
        plt.bar(x_values, dist_switch_matrix, label='switch number', color='green', linewidth=0.6)
        
        # Set the y-axis ticks at intervals of 0.5
        plt.xticks(np.linspace(1, 10 ,10))
        
        # Insert medians line
        plt.axvline(x=median, color='green', linestyle='--')
        
        # Inser text
        plt.text(median +0.05, max(dist_switch_matrix) + 0.01, 'median: ' + str(median), color='green')
        
        # Add a chart title
        plt.title(run + ' - time span: ' + str(t_span))
        
        # Set the y axis limits
        lower_limit = 0.0
        upper_limit = 0.5
        plt.ylim(lower_limit,upper_limit)
        
        # Add labels and legend
        plt.xlabel('Number of switches')
        plt.ylabel('Y Values')
        # plt.legend(fontsize=8)
        
        
        # Save the figure to a file (e.g., 'scatter_plot.png')
        plt.savefig(os.path.join(plot_dir, run + '_tspan' + str(t_span) + '_switch_number_distribution.pdf'), dpi=300)
        # plt.savefig(os.path.join(report_dir, 'overall_distribution' ,run + '_'+str(i)+ '_overall_dist_chart.pdf'), dpi=300)
        plt.show()
        
        # COMPUTE NUMBER OF SWITCHES ANALYSIS
        dist_switch_median = np.median(hist_array)
        dist_switch_5      = np.percentile(hist_array, 5)
        dist_switch_25     = np.percentile(hist_array, 25)
        dist_switch_75     = np.percentile(hist_array,75)
        dist_switch_95     = np.percentile(hist_array, 95)
        
        
        # PRINT SCOUR AND DEPOSITION MEDIAN
        print('Number of switch distribution')
        
        report = np.array([dist_switch_5, dist_switch_25, dist_switch_median, dist_switch_75, dist_switch_95])
        print('dist_switch_5, dist_switch_25, dist_switch_median, dist_switch_75, dist_switch_95')
        print(report)
        
        if 'poisson_dist' in plot_mode:
            
            
            data_set = hist_array
            
            # the bins have to be kept as a positive integer because poisson is a positive integer distribution
            bins = np.arange(10) + 0.5
            entries, bin_edges, patches = plt.hist(data_set, bins=bins, density=True, label="Data", width=0.8)
            
            # calculate bin centers
            middles_bins = (bin_edges[1:] + bin_edges[:-1]) * 0.5
            
            
            def fit_function(k, lamb):
                # The parameter lamb will be used as the fit parameter
                return poisson.pmf(k, lamb)
            
            
            # fit with curve_fit
            parameters, cov_matrix = curve_fit(fit_function, middles_bins, entries)
            
            # plot poisson-deviation with fitted parameter
            x_plot = np.arange(1, 10)
            
            plt.xticks(np.linspace(1, 10 ,10))
            
            plt.plot(
                x_plot,
                fit_function(x_plot, *parameters),
                marker="D",
                linestyle="-",
                color="red",
                label="Fit result",
            )
            
            # Add text in a box in the upper right corner
            text_content = 'Poisson parameter: ' + str(np.round(parameters,3))
            box_properties = dict(boxstyle='round', facecolor='white', edgecolor='black', alpha=0.7)
            
            plt.annotate(text_content, xy=(0.95, 0.75), xycoords='axes fraction',
                         fontsize=10, ha='right', va='top', bbox=box_properties)

            plt.title(run + ' _ Switch number - Poisson distribution fit')
            plt.legend()
            plt.show()