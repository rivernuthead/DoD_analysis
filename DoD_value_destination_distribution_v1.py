#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 14:56:12 2023

@author: erri

Aims:
    1. Scour and deposition dynamics and trajectories: cell by cell what happen
    a time t+1 to a pixel that at time t was i.e. scour?
    2. What is the probability for a scour pixel at time t to become a deppixel
    at time t+1?
    3. There are more likely to be trajectories for DoD values? 

Description:
this script takes as input the DoD stack for each discharge.
For each DoD, the script classify the DoD values in 10 classes (the classes
have been choosen manually to have more or less the same amount of values into
each class). Then the script create 10 array for each class where to store the
DoD value at the time t+1 of the DoD value at time t that belongs to a certain
class. So, the array class1 contains the values at the time t+1 of all the
values that at time t had belonged to the class1.
In this way, step by step, the distribution of the destination values is
provided. All the distribution for each dischare is stored in
step_by_step_dist_v2.odt spreadsheet.
It is also possible to clusterize these distribution and obtain a distribution
for all the runs. These charts are an output of this script, for each discharge.
Each line is a departure class and the x-axis of the distribution plot is the
destination value.
Classifying the destination values in the same way the departure values are it
is possible obtain bars charts where for each departure class the bars describe
at which class, in percentage, the arrival values belong.
"""


import numpy as np
import os
import matplotlib.pyplot as plt


script_name = os.path.basename(__file__)
print(script_name)

#%%
runs = ['q07_1', 'q10_2','q10_3','q10_4', 'q15_2','q15_3', 'q20_2']

# runs = ['q10_2','q10_3','q10_4', 'q15_2','q15_3', 'q20_2']

# runs = ['q07_1', 'q10_2', 'q15_2', 'q20_2']

# runs = ['q07_1', 'q10_2', 'q15_2']

# runs = ['q20_2']

# runs = ['q15_3']

# runs = ['q07_1']



for run in runs:
    delta = 1

    num_bins = 11 # For the departure classes
    num_bins_overall = 601
    bin_edges = [-60.0,-8.6,-5.8,-4.0,-2.6,-1.3,1.3,2.3,3.7,5.5,8.3,60.0]
    class_labels = [f"{bin_edges[i]},{bin_edges[i+1]}" for i in range(len(bin_edges) - 1)]
    hist_range = (-60, 60)  # Range of values to include in the histogram
    
    home_dir = os.getcwd()
    report_dir = os.path.join(home_dir, 'output', 'report_' + run)
    output_dir = os.path.join(report_dir, 'dist_step_by_step')
    if not(os.path.exists(output_dir)):
        os.mkdir(output_dir)
    if not(os.path.exists(os.path.join(report_dir, 'overall_distribution'))):
        os.mkdir(os.path.join(report_dir, 'overall_distribution'))
    

    plot_out_dir = os.path.join(home_dir, 'plot')
    
    data = np.load(os.path.join(home_dir,'output','DoDs', 'DoDs_stack', 'DoD_stack_'+run+'.npy'))
    

    
    timespan = 1
    
    data = data[:,:,:,timespan-1]
    
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
    
    data_sco = data*(data<0)
    data_dep = data*(data>0)
    
    # Initialize partail dist_class array
    # This is the matrix where the distribution at each timestep will be stored
    dist_array_sco_partial_class1_matrix = np.zeros((data.shape[0],num_bins))
    dist_array_sco_partial_class2_matrix = np.zeros((data.shape[0],num_bins))
    dist_array_sco_partial_class3_matrix = np.zeros((data.shape[0],num_bins))
    dist_array_sco_partial_class4_matrix = np.zeros((data.shape[0],num_bins))
    dist_array_sco_partial_class5_matrix = np.zeros((data.shape[0],num_bins))
    dist_array_dep_partial_class6_matrix = np.zeros((data.shape[0],num_bins))
    dist_array_dep_partial_class7_matrix = np.zeros((data.shape[0],num_bins))
    dist_array_dep_partial_class8_matrix = np.zeros((data.shape[0],num_bins))
    dist_array_dep_partial_class9_matrix = np.zeros((data.shape[0],num_bins))
    dist_array_dep_partial_class10_matrix = np.zeros((data.shape[0],num_bins))
    
    dist_array_uthrs_partial_class_matrix = np.zeros((data.shape[0],num_bins)) # This is the distribution of values from -2mm to +2mm

    # This is the array where the overall distribution will be stored
    dist_array_sco_class1 = []
    dist_array_sco_class2 = []
    dist_array_sco_class3 = []
    dist_array_sco_class4 = []
    dist_array_sco_class5 = []
    dist_array_dep_class6 = []
    dist_array_dep_class7 = []
    dist_array_dep_class8 = []
    dist_array_dep_class9 = []
    dist_array_dep_class10 = []

    dist_array_uthrs_class = [] # This is the distribution of values from -2mm to +2mm

    matrix = data
    for t in range(data.shape[0]-1):
        dist_array_sco_partial_class1 = []
        dist_array_sco_partial_class2 = []
        dist_array_sco_partial_class3 = []
        dist_array_sco_partial_class4 = []
        dist_array_sco_partial_class5 = []
        dist_array_uthrs_partial_class = []
        dist_array_dep_partial_class6 = []
        dist_array_dep_partial_class7 = []
        dist_array_dep_partial_class8 = []
        dist_array_dep_partial_class9 = []
        dist_array_dep_partial_class10 = []
        
        for x in range(data.shape[1]):
            for y in range(data.shape[2]):
                
                if bin_edges[0]<=matrix[t,x,y]<bin_edges[1]:
                    dist_array_sco_class1 = np.append(dist_array_sco_class1, data[t+1,x,y])
                    dist_array_sco_partial_class1 = np.append(dist_array_sco_partial_class1, data[t+1,x,y])
                if bin_edges[1]<=matrix[t,x,y]<bin_edges[2]:
                    dist_array_sco_class2 = np.append(dist_array_sco_class2, data[t+1,x,y])
                    dist_array_sco_partial_class2 = np.append(dist_array_sco_partial_class2, data[t+1,x,y])
                if bin_edges[2]<=matrix[t,x,y]<bin_edges[3]:
                    dist_array_sco_class3 = np.append(dist_array_sco_class3, data[t+1,x,y])
                    dist_array_sco_partial_class3 = np.append(dist_array_sco_partial_class3, data[t+1,x,y])
                if bin_edges[3]<=matrix[t,x,y]<bin_edges[4]:
                    dist_array_sco_class4 = np.append(dist_array_sco_class4, data[t+1,x,y])
                    dist_array_sco_partial_class4 = np.append(dist_array_sco_partial_class4, data[t+1,x,y])
                if bin_edges[4]<=matrix[t,x,y]<bin_edges[5]:
                    dist_array_sco_class5 = np.append(dist_array_sco_class5, data[t+1,x,y])
                    dist_array_sco_partial_class5 = np.append(dist_array_sco_partial_class5, data[t+1,x,y])
                if matrix[t,x,y]==0:
                    dist_array_uthrs_class = np.append(dist_array_uthrs_class, data[t+1,x,y])
                    dist_array_uthrs_partial_class = np.append(dist_array_uthrs_partial_class, data[t+1,x,y])
                if bin_edges[6]<matrix[t,x,y]<bin_edges[7]:
                    dist_array_dep_class6 = np.append(dist_array_dep_class6, data[t+1,x,y])
                    dist_array_dep_partial_class6 = np.append(dist_array_dep_partial_class6, data[t+1,x,y])
                if bin_edges[7]<=matrix[t,x,y]<bin_edges[8]:
                    dist_array_dep_class7 = np.append(dist_array_dep_class7, data[t+1,x,y])
                    dist_array_dep_partial_class7 = np.append(dist_array_dep_partial_class7, data[t+1,x,y])
                if bin_edges[8]<=matrix[t,x,y]<bin_edges[9]:
                    dist_array_dep_class8 = np.append(dist_array_dep_class8, data[t+1,x,y])
                    dist_array_dep_partial_class8 = np.append(dist_array_dep_partial_class8, data[t+1,x,y])
                if bin_edges[9]<=matrix[t,x,y]<bin_edges[10]:
                    dist_array_dep_class9 = np.append(dist_array_dep_class9, data[t+1,x,y])
                    dist_array_dep_partial_class9 = np.append(dist_array_dep_partial_class9, data[t+1,x,y])
                if bin_edges[10]<=matrix[t,x,y]<=bin_edges[11]:
                    dist_array_dep_class10 = np.append(dist_array_dep_class10, data[t+1,x,y])
                    dist_array_dep_partial_class10 = np.append(dist_array_dep_partial_class10, data[t+1,x,y])
                    
        hist_array = np.copy(dist_array_sco_partial_class1)
        hist_array = hist_array[~np.isnan(hist_array)] # Trim 0 and np.nan
        hist, bin_edges = np.histogram(hist_array, bins=bin_edges, range=hist_range)
        dist_array_sco_partial_class1_matrix[t,:] = hist/np.nansum(hist)
        
        hist_array = np.copy(dist_array_sco_partial_class2)
        hist_array = hist_array[~np.isnan(hist_array)] # Trim 0 and np.nan
        hist, bin_edges = np.histogram(hist_array, bins=bin_edges, range=hist_range)
        dist_array_sco_partial_class2_matrix[t,:] = hist/np.nansum(hist)
        
        hist_array = np.copy(dist_array_sco_partial_class3)
        hist_array = hist_array[~np.isnan(hist_array)] # Trim 0 and np.nan
        hist, bin_edges = np.histogram(hist_array, bins=bin_edges, range=hist_range)
        dist_array_sco_partial_class3_matrix[t,:] = hist/np.nansum(hist)
        
        hist_array = np.copy(dist_array_sco_partial_class4)
        hist_array = hist_array[~np.isnan(hist_array)] # Trim 0 and np.nan
        hist, bin_edges = np.histogram(hist_array, bins=bin_edges, range=hist_range)
        dist_array_sco_partial_class4_matrix[t,:] = hist/np.nansum(hist)
        
        hist_array = np.copy(dist_array_sco_partial_class5)
        hist_array = hist_array[~np.isnan(hist_array)] # Trim 0 and np.nan
        hist, bin_edges = np.histogram(hist_array, bins=bin_edges, range=hist_range)
        dist_array_sco_partial_class5_matrix[t,:] = hist/np.nansum(hist)
        
        hist_array = np.copy(dist_array_uthrs_partial_class)
        hist_array = hist_array[~np.isnan(hist_array)] # Trim 0 and np.nan
        hist, bin_edges = np.histogram(hist_array, bins=bin_edges, range=hist_range)
        dist_array_uthrs_partial_class_matrix[t,:] = hist/np.nansum(hist)
        
        hist_array = np.copy(dist_array_dep_partial_class6)
        hist_array = hist_array[~np.isnan(hist_array)] # Trim 0 and np.nan
        hist, bin_edges = np.histogram(hist_array, bins=bin_edges, range=hist_range)
        dist_array_dep_partial_class6_matrix[t,:] = hist/np.nansum(hist)
        
        hist_array = np.copy(dist_array_dep_partial_class7)
        hist_array = hist_array[~np.isnan(hist_array)] # Trim 0 and np.nan
        hist, bin_edges = np.histogram(hist_array, bins=bin_edges, range=hist_range)
        dist_array_dep_partial_class7_matrix[t,:] = hist/np.nansum(hist)
        
        hist_array = np.copy(dist_array_dep_partial_class8)
        hist_array = hist_array[~np.isnan(hist_array) & (hist_array!=0)] # Trim 0 and np.nan
        hist, bin_edges = np.histogram(hist_array, bins=bin_edges, range=hist_range)
        dist_array_dep_partial_class8_matrix[t,:] = hist/np.nansum(hist)
        
        hist_array = np.copy(dist_array_dep_partial_class9)
        hist_array = hist_array[~np.isnan(hist_array)] # Trim 0 and np.nan
        hist, bin_edges = np.histogram(hist_array, bins=bin_edges, range=hist_range)
        dist_array_dep_partial_class9_matrix[t,:] = hist/np.nansum(hist)
        
        hist_array = np.copy(dist_array_dep_partial_class10)
        hist_array = hist_array[~np.isnan(hist_array)] # Trim 0 and np.nan
        hist, bin_edges = np.histogram(hist_array, bins=bin_edges, range=hist_range)
        dist_array_dep_partial_class10_matrix[t,:] = hist/np.nansum(hist)
        
    # OVERALL DISTRIBUTION
    overall_dist_matrix = np.zeros((11,num_bins_overall))
    
    hist_array = np.copy(dist_array_sco_class1)
    hist_array = hist_array[~np.isnan(hist_array)] # Trim 0 and np.nan
    hist, bin_edges = np.histogram(hist_array, bins=num_bins_overall, range=hist_range)
    hist = hist/np.nansum(hist)
    # np.savetxt(os.path.join(report_dir, 'class1_distribution_dep.txt'), np.round(hist, decimals=2))
    overall_dist_matrix[0,:] = hist
    
    hist_array = np.copy(dist_array_sco_class2)
    hist_array = hist_array[~np.isnan(hist_array)] # Trim 0 and np.nan
    hist, bin_edges = np.histogram(hist_array, bins=num_bins_overall, range=hist_range)
    hist = hist/np.nansum(hist)
    # np.savetxt(os.path.join(report_dir, 'class2_distribution_dep.txt'), np.round(hist, decimals=2))
    overall_dist_matrix[1,:] = hist
    
    hist_array = np.copy(dist_array_sco_class3)
    hist_array = hist_array[~np.isnan(hist_array)] # Trim 0 and np.nan
    hist, bin_edges = np.histogram(hist_array, bins=num_bins_overall, range=hist_range)
    hist = hist/np.nansum(hist)
    # np.savetxt(os.path.join(report_dir, 'class3_distribution_dep.txt'), np.round(hist, decimals=2))
    overall_dist_matrix[2,:] = hist
    
    hist_array = np.copy(dist_array_sco_class4)
    hist_array = hist_array[~np.isnan(hist_array)] # Trim 0 and np.nan
    hist, bin_edges = np.histogram(hist_array, bins=num_bins_overall, range=hist_range)
    hist = hist/np.nansum(hist)
    # np.savetxt(os.path.join(report_dir, 'class4_distribution_dep.txt'), np.round(hist, decimals=2))
    overall_dist_matrix[3,:] = hist
    
    hist_array = np.copy(dist_array_sco_class5)
    hist_array = hist_array[~np.isnan(hist_array)] # Trim 0 and np.nan
    hist, bin_edges = np.histogram(hist_array, bins=num_bins_overall, range=hist_range)
    hist = hist/np.nansum(hist)
    # np.savetxt(os.path.join(report_dir, 'class5_distribution_dep.txt'), np.round(hist, decimals=2))
    overall_dist_matrix[4,:] = hist
    
    hist_array = np.copy(dist_array_uthrs_class)
    hist_array = hist_array[~np.isnan(hist_array)] # Trim 0 and np.nan
    hist, bin_edges = np.histogram(hist_array, bins=num_bins_overall, range=hist_range)
    hist = hist/np.nansum(hist)
    # np.savetxt(os.path.join(report_dir, 'class_uthrs_distribution.txt'), np.round(hist, decimals=2))
    overall_dist_matrix[5,:] = hist
    
    hist_array = np.copy(dist_array_dep_class6)
    hist_array = hist_array[~np.isnan(hist_array)] # Trim 0 and np.nan
    hist, bin_edges = np.histogram(hist_array, bins=num_bins_overall, range=hist_range)
    hist = hist/np.nansum(hist)
    # np.savetxt(os.path.join(report_dir, 'class7_distribution_dep.txt'), np.round(hist, decimals=2))
    overall_dist_matrix[6,:] = hist
    overall_dist_matrix
    hist_array = np.copy(dist_array_dep_class7)
    hist_array = hist_array[~np.isnan(hist_array)] # Trim 0 and np.nan
    hist, bin_edges = np.histogram(hist_array, bins=num_bins_overall, range=hist_range)
    hist = hist/np.nansum(hist)
    # np.savetxt(os.path.join(report_dir, 'class7_distribution_dep.txt'), np.round(hist, decimals=2))
    overall_dist_matrix[7,:] = hist
    
    hist_array = np.copy(dist_array_dep_class8)
    hist_array = hist_array[~np.isnan(hist_array)] # Trim 0 and np.nan
    hist, bin_edges = np.histogram(hist_array, bins=num_bins_overall, range=hist_range)
    hist = hist/np.nansum(hist)
    # np.savetxt(os.path.join(report_dir, 'class8_distribution_dep.txt'), np.round(hist, decimals=2))
    overall_dist_matrix[8,:] = hist
    
    hist_array = np.copy(dist_array_dep_class9)
    hist_array = hist_array[~np.isnan(hist_array)] # Trim 0 and np.nan
    hist, bin_edges = np.histogram(hist_array, bins=num_bins_overall, range=hist_range)
    hist = hist/np.nansum(hist)
    # np.savetxt(os.path.join(report_dir, 'class9_distribution_dep.txt'), np.round(hist, decimals=2))
    overall_dist_matrix[9,:] = hist
    
    hist_array = np.copy(dist_array_dep_class10)
    hist_array = hist_array[~np.isnan(hist_array)] # Trim 0 and np.nan
    hist, bin_edges = np.histogram(hist_array, bins=num_bins_overall, range=hist_range)
    hist = hist/np.nansum(hist)
    np.savetxt(os.path.join(report_dir, 'class10_distribution_dep.txt'), np.round(hist, decimals=2))
    overall_dist_matrix[10,:] = hist
    
    # np.savetxt(os.path.join(report_dir, run + '_overall_distribution_matrix.txt'), np.round(overall_dist_matrix, decimals=6))
    

#%%
    # GET DATA
    # overall_dist_matrix = np.loadtxt(os.path.join(report_dir, run + '_overall_distribution_matrix.txt'))    


    # PLOT THE OVERALL DISTRIBUTION
    
    # Generate x values ranging from -40 to 40
    num_bins_overall = 601
    x_values = np.linspace(-60, 60, num_bins_overall)
    
    # Create a color map with distinct colors for each row
    colors = plt.cm.viridis(np.linspace(0, 1, overall_dist_matrix.shape[0]))
    
    
    # Create a scatter plot for each row with different colors
    for i in range(overall_dist_matrix.shape[0]):
        data_hist = overall_dist_matrix[i, :]
        plt.plot(x_values, data_hist, label=class_labels[i], c=colors[i], linewidth=0.6)
        
        # Add a chart title
        plt.title(run + ' - Overall distribution')
        
        # Set the y axis limits
        lower_limit = 0
        upper_limit = 0.03
        plt.ylim(lower_limit,upper_limit)
        
        # Plot the peak value for x = 0
        # Find the index of x = 0 in the x_values array
        x_0_index = np.where(x_values == 0)[0][0]
        
        # Get the corresponding y-value for x = 0
        y_0 = data_hist[x_0_index]
        
        # Format the text with 4 decimal places
        text_label = '({:.3f})'.format(y_0)
        
        # Specify the exact coordinates for the text
        text_x = 12 # Adjust as needed
        text_y = upper_limit-0.002  # Adjust as needed
        
        # Add text at x = 0
        plt.text(text_x, text_y, text_label, ha='center', va='bottom')

        # Set x-axis labels
        # plt.xticks(x_values, class_labels, rotation=90)  # Rotate x-axis labels for readability
        
        # # Shading the area above the line plot
        # plt.fill_between(x_values[50:54], overall_dist_matrix[i,50:54], alpha=0.5)
        
        # Add labels and legend
        plt.xlabel('DoD destination values [mm]')
        plt.ylabel('Y Values')
        plt.legend(fontsize=8)
        
        # Save the figure to a file (e.g., 'scatter_plot.png')
        # plt.savefig(os.path.join(report_dir, 'overall_distribution' ,run + '_'+str(i)+'_'+class_labels[i]+ '_overall_dist_chart.pdf'), dpi=300)
        plt.savefig(os.path.join(report_dir, 'overall_distribution' ,run + '_'+str(i)+ '_overall_dist_chart.pdf'), dpi=300)
        # Show the plot (optional)
        plt.show()
    
    