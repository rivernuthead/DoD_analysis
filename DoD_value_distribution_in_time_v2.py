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
# runs = ['q07_1', 'q10_2', 'q15_2', 'q20_2']

# runs = ['q07_1', 'q10_2', 'q15_2']

runs = ['q20_2']

# runs = ['q07_1']

delta = 1

num_bins = 80
bin_edges = [-40.0,-10.0,-8.0,-6.0,-4.0,-2.0,2.0,4.0,6.0,8.0,10.0,40.0]
hist_range = (-40, 40)  # Range of values to include in the histogram

for run in runs:
    home_dir = os.getcwd()
    report_dir = os.path.join(home_dir, 'output', 'report_' + run)
    output_dir = os.path.join(report_dir, 'dist_step_by_step')
    if not(os.path.exists(output_dir)):
        os.mkdir(output_dir)
    plot_out_dir = os.path.join(home_dir, 'plot')
    
    data = np.load(os.path.join(home_dir,'output', 'DoDs', 'DoDs_stack', 'DoD_stack_'+run+'.npy'))
    
    timespan = 1
    
    data = data[:,:,:,timespan-1]
    
    # Create output matrix as below:
    #            t=1   t=2   t=3   t=4   t=5   t=6   t=7   t=8   t=9
    # delta = 1  1-0   2-1   3-2   4-3   5-4   6-5   7-6   8-7   9-8  average STDEV
    # delta = 2  2-0   3-1   4-2   5-3   6-4   7-5   8-6   9-7        average STDEV
    # delta = 3  3-0   4-1   5-2   6-3   7-4   8-5   9-6              average STDEV
    # delta = 4  4-0   5-1   6-2   7-3   8-4   9-5                    average STDEV
    # delta = 5  5-0   6-1   7-2   8-3   9-4                          average STDEV
    # delta = 6  6-0   7-1   8-2   9-3                                average STDEV
    # delta = 7  7-0   8-1   9-2                                      average STDEV
    # delta = 8  8-0   9-1                                            average STDEV
    # delta = 9  9-0                                                  average STDEV
    
    # stack[time, x, y, delta]
    
    data_sco = data*(data<0)
    data_dep = data*(data>0)
    
    # Initialize dist_class array
    
    dist_array_sco_partial_class1_matrix = np.zeros((9,num_bins))
    dist_array_sco_partial_class2_matrix = np.zeros((9,num_bins))
    dist_array_sco_partial_class3_matrix = np.zeros((9,num_bins))
    dist_array_sco_partial_class4_matrix = np.zeros((9,num_bins))
    dist_array_sco_partial_class5_matrix = np.zeros((9,num_bins))
    dist_array_dep_partial_class6_matrix = np.zeros((9,num_bins))
    dist_array_dep_partial_class7_matrix = np.zeros((9,num_bins))
    dist_array_dep_partial_class8_matrix = np.zeros((9,num_bins))
    dist_array_dep_partial_class9_matrix = np.zeros((9,num_bins))
    dist_array_dep_partial_class10_matrix = np.zeros((9,num_bins))

    dist_array_dep_class1 = []
    dist_array_dep_class2 = []
    dist_array_dep_class3 = []
    dist_array_dep_class4 = []
    dist_array_dep_class5 = []
    dist_array_dep_class6 = []
    dist_array_dep_class7 = []
    dist_array_dep_class8 = []
    dist_array_dep_class9 = []
    dist_array_dep_class10 = []
    
    dist_array_sco_class1 = []
    dist_array_sco_class2 = []
    dist_array_sco_class3 = []
    dist_array_sco_class4 = []
    dist_array_sco_class5 = []
    dist_array_sco_class6 = []
    dist_array_sco_class7 = []
    dist_array_sco_class8 = []
    dist_array_sco_class9 = []
    dist_array_sco_class10 = []


    # Loop over the spatio-temporal domain
    matrix = data_sco
    for t in range(data.shape[0]-1):
        dist_array_sco_partial_class1 = []
        dist_array_sco_partial_class2 = []
        dist_array_sco_partial_class3 = []
        dist_array_sco_partial_class4 = []
        dist_array_sco_partial_class5 = []
        for x in range(data.shape[1]):
            for y in range(data.shape[2]):
                if -40<=matrix[t,x,y]<-8:
                    dist_array_sco_class1 = np.append(dist_array_sco_class1, data[t+1,x,y])
                    dist_array_sco_partial_class1 = np.append(dist_array_sco_partial_class1, data[t+1,x,y])
                if -8<=matrix[t,x,y]<-5.7:
                    dist_array_sco_class2 = np.append(dist_array_sco_class2, data[t+1,x,y])
                    dist_array_sco_partial_class2 = np.append(dist_array_sco_partial_class2, data[t+1,x,y])
                if -5.7<=matrix[t,x,y]<-4.3:
                    dist_array_sco_class3 = np.append(dist_array_sco_class3, data[t+1,x,y])
                    dist_array_sco_partial_class3 = np.append(dist_array_sco_partial_class3, data[t+1,x,y])
                if -4.3<=matrix[t,x,y]<-3.3:
                    dist_array_sco_class4 = np.append(dist_array_sco_class4, data[t+1,x,y])
                    dist_array_sco_partial_class4 = np.append(dist_array_sco_partial_class4, data[t+1,x,y])
                if -3.3<=matrix[t,x,y]<0:
                    dist_array_sco_class5 = np.append(dist_array_sco_class5, data[t+1,x,y])
                    dist_array_sco_partial_class5 = np.append(dist_array_sco_partial_class5, data[t+1,x,y])
        
        hist_array = np.copy(dist_array_sco_partial_class1)
        hist_array = hist_array[~np.isnan(hist_array) & (hist_array!=0)] # Trim 0 and np.nan
        hist, bin_edges = np.histogram(hist_array, bins=num_bins, range=hist_range)
        dist_array_sco_partial_class1_matrix[t,:] = hist/np.nansum(hist)
        
        hist_array = np.copy(dist_array_sco_partial_class2)
        hist_array = hist_array[~np.isnan(hist_array) & (hist_array!=0)] # Trim 0 and np.nan
        hist, bin_edges = np.histogram(hist_array, bins=num_bins, range=hist_range)
        dist_array_sco_partial_class2_matrix[t,:] = hist/np.nansum(hist)
        
        hist_array = np.copy(dist_array_sco_partial_class3)
        hist_array = hist_array[~np.isnan(hist_array) & (hist_array!=0)] # Trim 0 and np.nan
        hist, bin_edges = np.histogram(hist_array, bins=num_bins, range=hist_range)
        dist_array_sco_partial_class3_matrix[t,:] = hist/np.nansum(hist)
        
        hist_array = np.copy(dist_array_sco_partial_class4)
        hist_array = hist_array[~np.isnan(hist_array) & (hist_array!=0)] # Trim 0 and np.nan
        hist, bin_edges = np.histogram(hist_array, bins=num_bins, range=hist_range)
        dist_array_sco_partial_class4_matrix[t,:] = hist/np.nansum(hist)
        
        hist_array = np.copy(dist_array_sco_partial_class5)
        hist_array = hist_array[~np.isnan(hist_array) & (hist_array!=0)] # Trim 0 and np.nan
        hist, bin_edges = np.histogram(hist_array, bins=num_bins, range=hist_range)
        dist_array_sco_partial_class5_matrix[t,:] = hist/np.nansum(hist)
        
        
    matrix = data_dep
    for t in range(data.shape[0]-1):
        dist_array_dep_partial_class6 = []
        dist_array_dep_partial_class7 = []
        dist_array_dep_partial_class8 = []
        dist_array_dep_partial_class9 = []
        dist_array_dep_partial_class10 = []
        for x in range(data.shape[1]):
            for y in range(data.shape[2]):

                if 0<=matrix[t,x,y]<2.5:
                    dist_array_dep_class6 = np.append(dist_array_dep_class6, data[t+1,x,y])
                    dist_array_dep_partial_class6 = np.append(dist_array_dep_partial_class6, data[t+1,x,y])
                if 2.5<=matrix[t,x,y]<3.3:
                    dist_array_dep_class7 = np.append(dist_array_dep_class7, data[t+1,x,y])
                    dist_array_dep_partial_class7 = np.append(dist_array_dep_partial_class7, data[t+1,x,y])
                if 3.3<=matrix[t,x,y]<4.7:
                    dist_array_dep_class8 = np.append(dist_array_dep_class8, data[t+1,x,y])
                    dist_array_dep_partial_class8 = np.append(dist_array_dep_partial_class8, data[t+1,x,y])
                if 4.7<=matrix[t,x,y]<6.7:
                    dist_array_dep_class9 = np.append(dist_array_dep_class9, data[t+1,x,y])
                    dist_array_dep_partial_class9 = np.append(dist_array_dep_partial_class9, data[t+1,x,y])
                if 6.7<=matrix[t,x,y]<=40:
                    dist_array_dep_class10 = np.append(dist_array_dep_class10, data[t+1,x,y])
                    dist_array_dep_partial_class10 = np.append(dist_array_dep_partial_class10, data[t+1,x,y])
                    
        hist_array = np.copy(dist_array_dep_partial_class6)
        hist_array = hist_array[~np.isnan(hist_array) & (hist_array!=0)] # Trim 0 and np.nan
        hist, bin_edges = np.histogram(hist_array, bins=num_bins, range=hist_range)
        dist_array_dep_partial_class6_matrix[t,:] = hist/np.nansum(hist)
        
        hist_array = np.copy(dist_array_dep_partial_class7)
        hist_array = hist_array[~np.isnan(hist_array) & (hist_array!=0)] # Trim 0 and np.nan
        hist, bin_edges = np.histogram(hist_array, bins=num_bins, range=hist_range)
        dist_array_dep_partial_class7_matrix[t,:] = hist/np.nansum(hist)
        
        hist_array = np.copy(dist_array_dep_partial_class8)
        hist_array = hist_array[~np.isnan(hist_array) & (hist_array!=0)] # Trim 0 and np.nan
        hist, bin_edges = np.histogram(hist_array, bins=num_bins, range=hist_range)
        dist_array_dep_partial_class8_matrix[t,:] = hist/np.nansum(hist)
        
        hist_array = np.copy(dist_array_dep_partial_class9)
        hist_array = hist_array[~np.isnan(hist_array) & (hist_array!=0)] # Trim 0 and np.nan
        hist, bin_edges = np.histogram(hist_array, bins=num_bins, range=hist_range)
        dist_array_dep_partial_class9_matrix[t,:] = hist/np.nansum(hist)
        
        hist_array = np.copy(dist_array_dep_partial_class10)
        hist_array = hist_array[~np.isnan(hist_array) & (hist_array!=0)] # Trim 0 and np.nan
        hist, bin_edges = np.histogram(hist_array, bins=num_bins, range=hist_range)
        dist_array_dep_partial_class10_matrix[t,:] = hist/np.nansum(hist)
        
        
    # OVERALL DISTRIBUTION
    overall_dist_matrix = np.zeros((10,num_bins))
    
    hist_array = np.copy(dist_array_sco_class1)
    hist_array = hist_array[~np.isnan(hist_array) & (hist_array!=0)] # Trim 0 and np.nan
    hist, bin_edges = np.histogram(hist_array, bins=num_bins, range=hist_range)
    hist = hist/np.nansum(hist)
    np.savetxt(os.path.join(report_dir, 'dist_step_by_step', 'class1_distribution_dep.txt'), np.round(hist, decimals=2))
    overall_dist_matrix[0,:] = hist
    
    hist_array = np.copy(dist_array_sco_class2)
    hist_array = hist_array[~np.isnan(hist_array) & (hist_array!=0)] # Trim 0 and np.nan
    hist, bin_edges = np.histogram(hist_array, bins=num_bins, range=hist_range)
    hist = hist/np.nansum(hist)
    np.savetxt(os.path.join(report_dir, 'dist_step_by_step', 'class2_distribution_dep.txt'), np.round(hist, decimals=2))
    overall_dist_matrix[1,:] = hist
    
    hist_array = np.copy(dist_array_sco_class3)
    hist_array = hist_array[~np.isnan(hist_array) & (hist_array!=0)] # Trim 0 and np.nan
    hist, bin_edges = np.histogram(hist_array, bins=num_bins, range=hist_range)
    hist = hist/np.nansum(hist)
    np.savetxt(os.path.join(report_dir, 'dist_step_by_step', 'class3_distribution_dep.txt'), np.round(hist, decimals=2))
    overall_dist_matrix[2,:] = hist
    
    hist_array = np.copy(dist_array_sco_class4)
    hist_array = hist_array[~np.isnan(hist_array) & (hist_array!=0)] # Trim 0 and np.nan
    hist, bin_edges = np.histogram(hist_array, bins=num_bins, range=hist_range)
    hist = hist/np.nansum(hist)
    np.savetxt(os.path.join(report_dir, 'dist_step_by_step', 'class4_distribution_dep.txt'), np.round(hist, decimals=2))
    overall_dist_matrix[3,:] = hist
    
    hist_array = np.copy(dist_array_sco_class5)
    hist_array = hist_array[~np.isnan(hist_array) & (hist_array!=0)] # Trim 0 and np.nan
    hist, bin_edges = np.histogram(hist_array, bins=num_bins, range=hist_range)
    hist = hist/np.nansum(hist)
    np.savetxt(os.path.join(report_dir, 'dist_step_by_step', 'class5_distribution_dep.txt'), np.round(hist, decimals=2))
    overall_dist_matrix[4,:] = hist
    
    hist_array = np.copy(dist_array_dep_class6)
    hist_array = hist_array[~np.isnan(hist_array) & (hist_array!=0)] # Trim 0 and np.nan
    hist, bin_edges = np.histogram(hist_array, bins=num_bins, range=hist_range)
    hist = hist/np.nansum(hist)
    np.savetxt(os.path.join(report_dir, 'dist_step_by_step', 'class6_distribution_dep.txt'), np.round(hist, decimals=2))
    overall_dist_matrix[5,:] = hist
    
    hist_array = np.copy(dist_array_dep_class7)
    hist_array = hist_array[~np.isnan(hist_array) & (hist_array!=0)] # Trim 0 and np.nan
    hist, bin_edges = np.histogram(hist_array, bins=num_bins, range=hist_range)
    hist = hist/np.nansum(hist)
    np.savetxt(os.path.join(report_dir, 'dist_step_by_step', 'class7_distribution_dep.txt'), np.round(hist, decimals=2))
    overall_dist_matrix[6,:] = hist
    
    hist_array = np.copy(dist_array_dep_class8)
    hist_array = hist_array[~np.isnan(hist_array) & (hist_array!=0)] # Trim 0 and np.nan
    hist, bin_edges = np.histogram(hist_array, bins=num_bins, range=hist_range)
    hist = hist/np.nansum(hist)
    np.savetxt(os.path.join(report_dir, 'dist_step_by_step', 'class8_distribution_dep.txt'), np.round(hist, decimals=2))
    overall_dist_matrix[7,:] = hist
    
    hist_array = np.copy(dist_array_dep_class9)
    hist_array = hist_array[~np.isnan(hist_array) & (hist_array!=0)] # Trim 0 and np.nan
    hist, bin_edges = np.histogram(hist_array, bins=num_bins, range=hist_range)
    hist = hist/np.nansum(hist)
    np.savetxt(os.path.join(report_dir, 'dist_step_by_step', 'class9_distribution_dep.txt'), np.round(hist, decimals=2))
    overall_dist_matrix[8,:] = hist
    
    hist_array = np.copy(dist_array_dep_class10)
    hist_array = hist_array[~np.isnan(hist_array) & (hist_array!=0)] # Trim 0 and np.nan
    hist, bin_edges = np.histogram(hist_array, bins=num_bins, range=hist_range)
    hist = hist/np.nansum(hist)
    np.savetxt(os.path.join(report_dir, 'dist_step_by_step', 'class10_distribution_dep.txt'), np.round(hist, decimals=2))
    overall_dist_matrix[9,:] = hist
    

#%%
    
    # PLOT THE OVERALL DISTRIBUTION
    
    # Generate x values ranging from -40 to 40
    x_values = np.linspace(-40, 40, num_bins)
    
    # Create a color map with distinct colors for each row
    colors = plt.cm.viridis(np.linspace(0, 1, overall_dist_matrix.shape[0]))
    
    # Labels
    labels = ['[-40.0,-8.0[',
              '[-8.0,-5.7[',
              '[-5.7,-4.3[',
              '[-4.3,-3.3[',
              '[-3.3,-2.0]',
              '[2.0,2.5[',
              '[2.5,3.3[',
              '[3.3,4.7[',
              '[4.7,6.7[',
              '[6.7,40.0]']
    
    # Create a scatter plot for each row with different colors
    for i in range(overall_dist_matrix.shape[0]):
        plt.plot(x_values, overall_dist_matrix[i, :], label=labels[i], c=colors[i], linewidth=0.6)
    
    # Add a chart title
    plt.title(run + ' - Overall distribution')
    
    # Set the y axis limits
    plt.ylim(0,0.18)
    
    # # Shading the area above the line plot
    # plt.fill_between(x_values[50:54], overall_dist_matrix[i,50:54], alpha=0.5)
    
    # Add labels and legend
    plt.xlabel('DoD destination values [mm]')
    plt.ylabel('Y Values')
    plt.legend(fontsize=8)
    
    # Save the figure to a file (e.g., 'scatter_plot.png')
    plt.savefig(os.path.join(report_dir, 'dist_step_by_step', run + 'overall_dist_chart.pdf'), dpi=300)
    
    # Show the plot (optional)
    plt.show()
    
    
#%%
    # Classify destination frequency distribution
    
    dest_integral = []
    classes_lim = [-40.0,-8.0,-5.7,-4.3,-3.3,-2.0,2.5,3.3,4.7,6.7,40.0]
    
    for i in range(overall_dist_matrix.shape[0]):
        dest_classes = []
        for j in range(0, len(classes_lim)-1):
            lim_inf = classes_lim[j]
            lim_sup = classes_lim[j+1]
    
            mask = (x_values>=lim_inf)*(x_values<lim_sup)
            overall_dist_mkd = overall_dist_matrix[i,:]*mask
        
            dest_classes = np.append(dest_classes, np.nansum(overall_dist_mkd))
        # print(dest_classes, ' - Array sum: ', np.nansum(dest_classes))
        
        # Create x values for the bars (you can use integer indices as x positions)
        x_data = np.array(np.linspace(1,10,10), dtype=np.int8)
        
        # Create a vertical bar plot
        plt.bar(x_data, dest_classes)
        
        # Optionally, add labels to the x-axis
        plt.xticks(x_data, x_data)
        
        # Add values above each bar
        for x, y in zip(x_data, dest_classes):
            plt.text(x, y, str(np.round(y, decimals=3)), ha='center', va='bottom')
            
            
        plt.ylim(0,0.50)
        
        
        # Add labels and title to the plot
        plt.xlabel('X Values')
        plt.ylabel('Y Values')
        plt.title(run + ' - Class: ' + str(i))
        
        # Save the figure to a file (e.g., 'scatter_plot.png')
        plt.savefig(os.path.join(report_dir, 'dist_step_by_step', run + '_' + str(i) + '_overall_dest_probability.pdf'), dpi=300)
        
        # Show the plot (optional)
        plt.show()
        
    
#%%
    
    # SAVE THE MATRIX WITH THE 9 DISTRIBUTION, ONE FOR EACH TIMESTEP
    np.savetxt(os.path.join(report_dir, 'dist_step_by_step', 'partial_class1_distribution_sco.txt'), np.round(dist_array_sco_partial_class1_matrix, decimals=2), delimiter=',')
    np.savetxt(os.path.join(report_dir, 'dist_step_by_step', 'partial_class2_distribution_sco.txt'), np.round(dist_array_sco_partial_class2_matrix, decimals=2), delimiter=',')
    np.savetxt(os.path.join(report_dir, 'dist_step_by_step', 'partial_class3_distribution_sco.txt'), np.round(dist_array_sco_partial_class3_matrix, decimals=2), delimiter=',')
    np.savetxt(os.path.join(report_dir, 'dist_step_by_step', 'partial_class4_distribution_sco.txt'), np.round(dist_array_sco_partial_class4_matrix, decimals=2), delimiter=',')
    np.savetxt(os.path.join(report_dir, 'dist_step_by_step', 'partial_class5_distribution_sco.txt'), np.round(dist_array_sco_partial_class5_matrix, decimals=2), delimiter=',')
    
    np.savetxt(os.path.join(report_dir, 'dist_step_by_step', 'partial_class6_distribution_dep.txt'), np.round(dist_array_dep_partial_class6_matrix, decimals=2), delimiter=',')
    np.savetxt(os.path.join(report_dir, 'dist_step_by_step', 'partial_class7_distribution_dep.txt'), np.round(dist_array_dep_partial_class7_matrix, decimals=2), delimiter=',')
    np.savetxt(os.path.join(report_dir, 'dist_step_by_step', 'partial_class8_distribution_dep.txt'), np.round(dist_array_dep_partial_class8_matrix, decimals=2), delimiter=',')
    np.savetxt(os.path.join(report_dir, 'dist_step_by_step', 'partial_class9_distribution_dep.txt'), np.round(dist_array_dep_partial_class9_matrix, decimals=2), delimiter=',')
    np.savetxt(os.path.join(report_dir, 'dist_step_by_step', 'partial_class10_distribution_dep.txt'), np.round(dist_array_dep_partial_class10_matrix, decimals=2), delimiter=',')
    
