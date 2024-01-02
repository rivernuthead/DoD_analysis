#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 14:56:12 2023

@author: erri

Aims:
    1. Scour and deposition dynamics and trajectories: cell by cell what happen
    at time t+1 to a pixel that at time t was i.e. scour?
    2. What is the probability for a scour pixel at time t to become a dep 
    pixel at time t+1?
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

delta = 1

num_bins = 11
bin_edges = [-60.0,-8.0,-5.8,-4.2,-3.2,-2.0,2.0,2.8,3.8,5.0,7.0,60.0]
class_labels = [f"{bin_edges[i]},{bin_edges[i+1]}" for i in range(len(bin_edges) - 1)]
hist_range = (-40, 40)  # Range of values to include in the histogram

for run in runs:
    home_dir = os.getcwd()
    report_dir = os.path.join(home_dir, 'output', 'report_' + run)
    output_dir = os.path.join(report_dir, 'dist_step_by_step')
    if not(os.path.exists(output_dir)):
        os.mkdir(output_dir)

    plot_out_dir = os.path.join(home_dir, 'plot')
    
    data = np.load(os.path.join(home_dir,'output','DoDs', 'DoDs_stack', 'DoD_stack_'+run+'.npy'))
    

    for timespan in range(1,data.shape[3]):
        data = np.load(os.path.join(home_dir,'output','DoDs', 'DoDs_stack', 'DoD_stack_'+run+'.npy'))
        
        if timespan ==1:
            data = data[:,:,:,timespan-1]
            print('Timespan: ' + str(timespan-1))
        else:
            data = data[:-timespan+1,:,:,timespan-1]
            print('Timespan: :' + str(-timespan+1))
        
        '''
        DoD input stack structure:
                    t=1   t=2   t=3   t=4   t=5   t=6   t=7   t=8   t=9
        delta = 1   1-0   2-1   3-2   4-3   5-4   6-5   7-6   8-7   9-8
        delta = 2   2-0   3-1   4-2   5-3   6-4   7-5   8-6   9-7
        delta = 3   3-0   4-1   5-2   6-3   7-4   8-5   9-6
        delta = 4   4-0   5-1   6-2   7-3   8-4   9-5
        delta = 5   5-0   6-1   7-2   8-3   9-4
        delta = 6   6-0   7-1   8-2   9-3
        delta = 7   7-0   8-1   9-2
        delta = 8   8-0   9-1
        delta = 9   9-0
        
        stack[time, x, y, delta]
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
        overall_dist_matrix = np.zeros((11,num_bins))
        
        hist_array = np.copy(dist_array_sco_class1)
        hist_array = hist_array[~np.isnan(hist_array)] # Trim 0 and np.nan
        hist, bin_edges = np.histogram(hist_array, bins=bin_edges, range=hist_range)
        hist = hist/np.nansum(hist)
        np.savetxt(os.path.join(report_dir, 'class1_distribution_dep_timespan'+str(timespan) + '.txt'), np.round(hist, decimals=2))
        overall_dist_matrix[0,:] = hist
        
        hist_array = np.copy(dist_array_sco_class2)
        hist_array = hist_array[~np.isnan(hist_array)] # Trim 0 and np.nan
        hist, bin_edges = np.histogram(hist_array, bins=bin_edges, range=hist_range)
        hist = hist/np.nansum(hist)
        np.savetxt(os.path.join(report_dir, 'class2_distribution_dep_timespan'+str(timespan) + '.txt'), np.round(hist, decimals=2))
        overall_dist_matrix[1,:] = hist
        
        hist_array = np.copy(dist_array_sco_class3)
        hist_array = hist_array[~np.isnan(hist_array)] # Trim 0 and np.nan
        hist, bin_edges = np.histogram(hist_array, bins=bin_edges, range=hist_range)
        hist = hist/np.nansum(hist)
        np.savetxt(os.path.join(report_dir, 'class3_distribution_dep_timespan'+str(timespan) + '.txt'), np.round(hist, decimals=2))
        overall_dist_matrix[2,:] = hist
        
        hist_array = np.copy(dist_array_sco_class4)
        hist_array = hist_array[~np.isnan(hist_array)] # Trim 0 and np.nan
        hist, bin_edges = np.histogram(hist_array, bins=bin_edges, range=hist_range)
        hist = hist/np.nansum(hist)
        np.savetxt(os.path.join(report_dir, 'class4_distribution_dep_timespan'+str(timespan) + '.txt'), np.round(hist, decimals=2))
        overall_dist_matrix[3,:] = hist
        
        hist_array = np.copy(dist_array_sco_class5)
        hist_array = hist_array[~np.isnan(hist_array)] # Trim 0 and np.nan
        hist, bin_edges = np.histogram(hist_array, bins=bin_edges, range=hist_range)
        hist = hist/np.nansum(hist)
        np.savetxt(os.path.join(report_dir, 'class5_distribution_dep_timespan'+str(timespan) + '.txt'), np.round(hist, decimals=2))
        overall_dist_matrix[4,:] = hist
        
        #TODO
        hist_array = np.copy(dist_array_uthrs_class)
        hist_array = hist_array[~np.isnan(hist_array)] # Trim 0 and np.nan
        hist, bin_edges = np.histogram(hist_array, bins=bin_edges, range=hist_range)
        hist = hist/np.nansum(hist)
        np.savetxt(os.path.join(report_dir, 'class_uthrs_distribution_timespan'+str(timespan) + '.txt'), np.round(hist, decimals=2))
        overall_dist_matrix[5,:] = hist
        
        hist_array = np.copy(dist_array_dep_class6)
        hist_array = hist_array[~np.isnan(hist_array)] # Trim 0 and np.nan
        hist, bin_edges = np.histogram(hist_array, bins=bin_edges, range=hist_range)
        hist = hist/np.nansum(hist)
        np.savetxt(os.path.join(report_dir, 'class7_distribution_dep_timespan'+str(timespan) +'.txt'), np.round(hist, decimals=2))
        overall_dist_matrix[6,:] = hist
        overall_dist_matrix
        hist_array = np.copy(dist_array_dep_class7)
        hist_array = hist_array[~np.isnan(hist_array)] # Trim 0 and np.nan
        hist, bin_edges = np.histogram(hist_array, bins=bin_edges, range=hist_range)
        hist = hist/np.nansum(hist)
        np.savetxt(os.path.join(report_dir, 'class7_distribution_dep_timespan'+str(timespan) +'.txt'), np.round(hist, decimals=2))
        overall_dist_matrix[7,:] = hist
        
        hist_array = np.copy(dist_array_dep_class8)
        hist_array = hist_array[~np.isnan(hist_array)] # Trim 0 and np.nan
        hist, bin_edges = np.histogram(hist_array, bins=bin_edges, range=hist_range)
        hist = hist/np.nansum(hist)
        np.savetxt(os.path.join(report_dir, 'class8_distribution_dep_timespan'+str(timespan) +'.txt'), np.round(hist, decimals=2))
        overall_dist_matrix[8,:] = hist
        
        hist_array = np.copy(dist_array_dep_class9)
        hist_array = hist_array[~np.isnan(hist_array)] # Trim 0 and np.nan
        hist, bin_edges = np.histogram(hist_array, bins=bin_edges, range=hist_range)
        hist = hist/np.nansum(hist)
        np.savetxt(os.path.join(report_dir, 'class9_distribution_dep_timespan'+str(timespan) +'.txt'), np.round(hist, decimals=2))
        overall_dist_matrix[9,:] = hist
        
        hist_array = np.copy(dist_array_dep_class10)
        hist_array = hist_array[~np.isnan(hist_array)] # Trim 0 and np.nan
        hist, bin_edges = np.histogram(hist_array, bins=bin_edges, range=hist_range)
        hist = hist/np.nansum(hist)
        np.savetxt(os.path.join(report_dir, 'class10_distribution_dep_timespan'+str(timespan) +'.txt'), np.round(hist, decimals=2))
        overall_dist_matrix[10,:] = hist
        
        np.savetxt(os.path.join(report_dir, run + '_overall_distribution_matrix_timespan'+str(timespan) +'.txt'), np.round(overall_dist_matrix, decimals=6))
        
    
    #%%
        # # GET DATA
        # # overall_dist_matrix = np.loadtxt(os.path.join(report_dir, run + '_overall_distribution_matrix_timespan'+str(timespan) +'.txt'))    
    
    
        # # PLOT THE OVERALL DISTRIBUTION
        
        # # Generate x values ranging from -40 to 40
        # num_bins_overall = 11
        # x_values = np.linspace(-40, 40, num_bins_overall)
        
        # # Create a color map with distinct colors for each row
        # colors = plt.cm.viridis(np.linspace(0, 1, overall_dist_matrix.shape[0]))
        
        
        # # Create a scatter plot for each row with different colors
        # for i in range(overall_dist_matrix.shape[0]):
        #     plt.plot(x_values, overall_dist_matrix[i, :], label=class_labels[i], c=colors[i], linewidth=0.6)
        
        # # Add a chart title
        # plt.title(run + ' - Overall distribution - Timespan: ' + str(timespan))
        
        # # Set x-axis labels
        # plt.xticks(x_values, class_labels, rotation=90)  # Rotate x-axis labels for readability
        
        # # Set the y axis limits
        # # plt.ylim(0,0.08)
        
        # # # Shading the area above the line plot
        # # plt.fill_between(x_values[50:54], overall_dist_matrix[i,50:54], alpha=0.5)
        
        # # Add labels and legend
        # plt.xlabel('DoD destination values [mm]')
        # plt.ylabel('Y Values')
        # plt.legend(fontsize=8)
        
        # # Save the figure to a file (e.g., 'scatter_plot.png')
        # plt.savefig(os.path.join(report_dir, run + '_overall_dist_chart_timespan'+ str(timespan)+'.pdf'), dpi=300)
        
        # # Show the plot (optional)
        # plt.show()
        
        
    #%%
        for i in range(overall_dist_matrix.shape[0]):
            
            # Define the data
            data = overall_dist_matrix[i,:]
            
            fig, ax = plt.subplots(figsize=(8, 6))
            
            # Plot the bars
            ax.bar(range(len(data)), data)
            
            
            # Set x axes ticks
            plt.xticks(range(len(data)), class_labels, rotation=90)  # Rotate x-axis labels for readability
    
            # Add values above each bar
            
            for x, y in zip(range(len(data)), data):
                if np.logical_not(np.isnan(y)):
                    plt.text(x, y, str(np.round(y, decimals=3)), ha='center', va='bottom')
             
            # Set the y limit
            # plt.ylim(0,0.50)
            plt.ylim(0,1.0)
            
            # Add labels and title to the plot
            # plt.xlabel('X Values')
            # plt.ylabel('Y Values')
            plt.title(run + ' - Class interval: ' + class_labels[i] + ' - Timespan: ' + str(timespan-1))
            # plt.xticks(bin_edges)
            # Save the figure to a file (e.g., 'scatter_plot.png')
            plt.savefig(os.path.join(report_dir, run + '_' + str(i) + '_overall_dest_probability_timespan' + str(timespan)+'.pdf'), dpi=300)
            
            
            # Show the plot
            plt.show()
            
    #%%
        # GET DATA
        overall_dist_matrix = np.loadtxt(os.path.join(report_dir, run + '_overall_distribution_matrix_timespan'+str(timespan)+ '.txt'))  
        
        
        sco = np.nansum(overall_dist_matrix[:,:5], axis=1)
        null = np.nansum(overall_dist_matrix[:,5:6], axis=1)
        dep = np.nansum(overall_dist_matrix[:,6:], axis=1)
        
        # Your data
        data1 = sco
        data2 = null
        data3 = dep
        
        # Bar width and alpha
        bar_width = 0.9
        alpha = 0.6
    
        # Create a bar plot with stacked values and different colors
        bars1 = plt.bar(range(len(data1)), data1, width=bar_width, alpha=alpha, label='Scour', color='r')
        bars2 = plt.bar(range(len(data2)), data2, width=bar_width, alpha=alpha, bottom=data1, label='N', color='g')
        bars3 = plt.bar(range(len(data3)), data3, width=bar_width, alpha=alpha, bottom=data1 + data2, label='Dep', color='b')
    
        # Set x-axis labels
        plt.xticks(range(len(data1)), class_labels, rotation=90)  # Rotate x-axis labels for readability
    
        # Set a title
        plt.title('Stacked Bar Chart - ' + run + ' - Timespan: ' + str(timespan-1))
    
        # Label the axes
        plt.xlabel('X-Axis Label')
        plt.ylabel('Y-Axis Label')
    
        # Add a legend
        plt.legend()
    
        # Add data labels with values centered within the bars
        for bar1, bar2, bar3 in zip(bars1, bars2, bars3):
            h1 = bar1.get_height()
            h2 = bar2.get_height()
            h3 = bar3.get_height()
            plt.text(
                bar1.get_x() + bar1.get_width() / 2, h1 / 2, str(np.round(h1, decimals=3)), ha='center', va='center', color='black')
            plt.text(
                bar2.get_x() + bar2.get_width() / 2, h1 + h2 / 2, str(np.round(h2, decimals=3)), ha='center', va='center', color='black')
            plt.text(
                bar3.get_x() + bar3.get_width() / 2, h1 + h2 + h3 / 2, str(np.round(h3, decimals=3)), ha='center', va='center', color='black')
            
        plt.savefig(os.path.join(report_dir, 'dist_step_by_step', run + '_overall_didstribution_SCO_FILL_timespan' + str(timespan)+'.pdf'), dpi=300)
        
        # Show the plot
        plt.show()
    
        
    #%%
        
        # SAVE THE MATRIX WITH THE 9 DISTRIBUTION, ONE FOR EACH TIMESTEP
        np.savetxt(os.path.join(report_dir, 'partial_class1_distribution_sco_timespan'+str(timespan)+ '.txt'), np.round(dist_array_sco_partial_class1_matrix, decimals=2), delimiter=',')
        np.savetxt(os.path.join(report_dir, 'partial_class2_distribution_sco_timespan'+str(timespan) +'.txt'), np.round(dist_array_sco_partial_class2_matrix, decimals=2), delimiter=',')
        np.savetxt(os.path.join(report_dir, 'partial_class3_distribution_sco_timespan'+str(timespan) +'.txt'), np.round(dist_array_sco_partial_class3_matrix, decimals=2), delimiter=',')
        np.savetxt(os.path.join(report_dir, 'partial_class4_distribution_sco_timespan'+str(timespan) +'.txt'), np.round(dist_array_sco_partial_class4_matrix, decimals=2), delimiter=',')
        np.savetxt(os.path.join(report_dir, 'partial_class5_distribution_sco_timespan'+str(timespan) +'.txt'), np.round(dist_array_sco_partial_class5_matrix, decimals=2), delimiter=',')
        
        np.savetxt(os.path.join(report_dir, 'partial_class_uthrs_distribution_timespan'+str(timespan) +'.txt'), np.round(dist_array_uthrs_partial_class_matrix, decimals=2), delimiter=',')
        
        np.savetxt(os.path.join(report_dir, 'partial_class6_distribution_dep_timespan'+str(timespan) +'.txt'), np.round(dist_array_dep_partial_class6_matrix, decimals=2), delimiter=',')
        np.savetxt(os.path.join(report_dir, 'partial_class7_distribution_dep_timespan'+str(timespan)+ '.txt'), np.round(dist_array_dep_partial_class7_matrix, decimals=2), delimiter=',')
        np.savetxt(os.path.join(report_dir, 'partial_class8_distribution_dep_timespan'+str(timespan) +'.txt'), np.round(dist_array_dep_partial_class8_matrix, decimals=2), delimiter=',')
        np.savetxt(os.path.join(report_dir, 'partial_class9_distribution_dep_timespan'+str(timespan) +'.txt'), np.round(dist_array_dep_partial_class9_matrix, decimals=2), delimiter=',')
        np.savetxt(os.path.join(report_dir, 'partial_class10_distribution_dep_timespan'+str(timespan)+ '.txt'), np.round(dist_array_dep_partial_class10_matrix, decimals=2), delimiter=',')
        
