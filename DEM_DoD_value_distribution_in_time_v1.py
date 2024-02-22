#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 14:56:12 2023

@author: erri

Aims:
    1. pots stacked bars chart with the class percentage of pixels that ens up
        in Scour, Deposition or No-changes.
    2. pots bars chart with the class percentage of pixels that ens up
        in Scour, Deposition or No-changes.

Description:

"""


import numpy as np
import os
import matplotlib.pyplot as plt


#%%
# runs = ['q07_1', 'q10_2','q10_3','q10_4', 'q15_2','q15_3', 'q20_2']

# runs = ['q10_2','q10_3','q10_4', 'q15_2','q15_3', 'q20_2']

runs = ['q07_1', 'q10_2', 'q15_3', 'q20_2']

# runs = ['q07_1', 'q10_2', 'q15_2']

# runs = ['q20_2']

# runs = ['q15_3']

# runs = ['q07_1']

# delta = 1

# Plot mode
plot_mode = [
    'single_class_distribution',
             'overall_class_distribution_stacked',
              'overall_class_distribution_bars'
             ]
# ANALYSIS PARAMETERS
jumps = np.array([1,2,3,4,5,6,7,8,9])

for jump in jumps:
    

    # DISTRIBUTION PARAMETERS
    num_bins = 7
    # bin_edges = [-50.0,-11.52,-4.59,-0.58,2.345,4.535,7.09,20.0]
    DEM_bin_edges = [-50.0,-12.62,-5.69,-1.67,1.25,3.805,6.36,20.0]
    # DoD_bid_edges = [-60.0,-8.6,-5.8,-4.0,-2.6,-1.3,1.3,2.3,3.7,5.5,8.3,60.0]
    DoD_bin_edges = [-60.0,-5.8,-2.6,-1.3,1.3,2.3,5.5,60.0]
    DEM_class_labels = [f"{DEM_bin_edges[i]},{DEM_bin_edges[i+1]}" for i in range(len(DEM_bin_edges) - 1)]
    DoD_class_labels = [f"{DoD_bin_edges[i]},{DoD_bin_edges[i+1]}" for i in range(len(DoD_bin_edges) - 1)]
    hist_range = (-50, 20)  # Range of values to include in the histogram
    
    for run in runs:
        print(run, ' is running...')
        print('Jump: ', jump)
        print()
        home_dir = os.getcwd()
        output_dir = os.path.join(home_dir, 'output', run, 'DEM_analysis')
        if not(os.path.exists(output_dir)):
            os.mkdir(output_dir)
            
        output_dir = os.path.join(home_dir, 'output', run, 'DEM_analysis', 'distribution_analysis')
        if not(os.path.exists(output_dir)):
            os.mkdir(output_dir)
        
        plot_dir = os.path.join(output_dir, 'polts')
        if not(os.path.exists(plot_dir)):
            os.mkdir(plot_dir)
        
        # LOAD DEM
        DEM_path = os.path.join(home_dir, 'surveys',run)
        DEM_stack_path = os.path.join(DEM_path, run + '_DEM_stack.npy')
        DEM_raw = np.load(DEM_stack_path)
        DEM_raw = np.where(DEM_raw==-999, np.nan, DEM_raw)
        dim_t, dim_y, dim_x = DEM_raw.shape
        
        # LOAD DoD
        '''
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
        DoD_stack_path = os.path.join(home_dir,'output','DoDs', 'DoDs_stack', 'DoD_stack_'+run+'.npy')
        DoD_raw = np.load(DoD_stack_path)
        DoD_data = DoD_raw[:,:,:,0]
        
        # path_report_pdf = os.path.join(output_dir)
        
        # for timespan in range(1,2):
        # for timespan in range(0,(data_raw.shape[3]//2)):
        # for timespan in range(0,(data_raw.shape[3]//2)):
        # print('Timespan: ', timespan)
        
        
        # print('Jump: ', jump)
        # data = data_raw[:,:,:,timespan]
        
        array = np.linspace(0,DEM_raw.shape[0]-1, DEM_raw.shape[0])
        
        array = array.astype(int)
        
        # t_max = DEM_raw.shape[0]
        
        DEM_data = np.copy(DEM_raw)
        
        for t in (range(0,DEM_raw.shape[0]-jump)):
            # t = t*jump
            # if t<=t_max-jump:
    
            data_sco = DoD_data*(DoD_data<0)
            data_dep = DoD_data*(DoD_data>0)
            
            # Initialize partail dist_class array
            # This is the matrix where the distribution at each timestep will be stored
            dist_array_sco_partial_class1_matrix = np.zeros((DEM_data.shape[0],num_bins))
            dist_array_sco_partial_class2_matrix = np.zeros((DEM_data.shape[0],num_bins))
            dist_array_sco_partial_class3_matrix = np.zeros((DEM_data.shape[0],num_bins))
            dist_array_sco_partial_class4_matrix = np.zeros((DEM_data.shape[0],num_bins))
            dist_array_sco_partial_class5_matrix = np.zeros((DEM_data.shape[0],num_bins))
            dist_array_dep_partial_class6_matrix = np.zeros((DEM_data.shape[0],num_bins))
            dist_array_dep_partial_class7_matrix = np.zeros((DEM_data.shape[0],num_bins))
            # dist_array_dep_partial_class8_matrix = np.zeros((data.shape[0],num_bins))
            # dist_array_dep_partial_class9_matrix = np.zeros((data.shape[0],num_bins))
            # dist_array_dep_partial_class10_matrix = np.zeros((data.shape[0],num_bins))
            
            # dist_array_uthrs_partial_class_matrix = np.zeros((data.shape[0],num_bins)) # This is the distribution of values from -2mm to +2mm
        
            # This is the array where the overall distribution will be stored
            dist_array_sco_class1 = []
            dist_array_sco_class2 = []
            dist_array_sco_class3 = []
            dist_array_sco_class4 = []
            dist_array_sco_class5 = []
            dist_array_dep_class6 = []
            dist_array_dep_class7 = []
            # dist_array_dep_class8 = []
            # dist_array_dep_class9 = []
            # dist_array_dep_class10 = []
        
            # dist_array_uthrs_class = [] # This is the distribution of values from -2mm to +2mm
            
            
            
            matrix = DEM_data
            # for t in range(data.shape[0]-jump):
            dist_array_sco_partial_class1 = []
            dist_array_sco_partial_class2 = []
            dist_array_sco_partial_class3 = []
            dist_array_sco_partial_class4 = []
            dist_array_sco_partial_class5 = []
            # dist_array_uthrs_partial_class = []
            dist_array_dep_partial_class6 = []
            dist_array_dep_partial_class7 = []
            # dist_array_dep_partial_class8 = []
            # dist_array_dep_partial_class9 = []
            # dist_array_dep_partial_class10 = []
            
            for x in range(DEM_data.shape[1]):
                for y in range(DEM_data.shape[2]):
                    
                    if DEM_bin_edges[0]<=matrix[t,x,y]<DEM_bin_edges[1]:
                        dist_array_sco_class1 = np.append(dist_array_sco_class1, DoD_data[t+jump-1,x,y])
                        dist_array_sco_partial_class1 = np.append(dist_array_sco_partial_class1, DoD_data[t+jump-1,x,y])
                    if DEM_bin_edges[1]<=matrix[t,x,y]<DEM_bin_edges[2]:
                        dist_array_sco_class2 = np.append(dist_array_sco_class2, DoD_data[t+jump-1,x,y])
                        dist_array_sco_partial_class2 = np.append(dist_array_sco_partial_class2, DoD_data[t+jump-1,x,y])
                    if DEM_bin_edges[2]<=matrix[t,x,y]<DEM_bin_edges[3]:
                        dist_array_sco_class3 = np.append(dist_array_sco_class3, DoD_data[t+jump-1,x,y])
                        dist_array_sco_partial_class3 = np.append(dist_array_sco_partial_class3, DoD_data[t+jump-1,x,y])
                    if DEM_bin_edges[3]<=matrix[t,x,y]<DEM_bin_edges[4]:
                        dist_array_sco_class4 = np.append(dist_array_sco_class4, DoD_data[t+jump-1,x,y])
                        dist_array_sco_partial_class4 = np.append(dist_array_sco_partial_class4, DoD_data[t+jump-1,x,y])
                    if DEM_bin_edges[4]<=matrix[t,x,y]<DEM_bin_edges[5]:
                        dist_array_sco_class5 = np.append(dist_array_sco_class5, DoD_data[t+jump-1,x,y])
                        dist_array_sco_partial_class5 = np.append(dist_array_sco_partial_class5, DoD_data[t+jump-1,x,y])
                    if DEM_bin_edges[5]<matrix[t,x,y]<DEM_bin_edges[6]:
                        dist_array_dep_class6 = np.append(dist_array_dep_class6, DoD_data[t+jump-1,x,y])
                        dist_array_dep_partial_class6 = np.append(dist_array_dep_partial_class6, DoD_data[t+jump-1,x,y])
                    if DEM_bin_edges[6]<=matrix[t,x,y]<DEM_bin_edges[7]:
                        dist_array_dep_class7 = np.append(dist_array_dep_class7, DoD_data[t+jump-1,x,y])
                        dist_array_dep_partial_class7 = np.append(dist_array_dep_partial_class7, DoD_data[t+jump-1,x,y])
                        
            hist_array = np.copy(dist_array_sco_partial_class1)
            hist_array = hist_array[~np.isnan(hist_array)] # Trim 0 and np.nan
            hist, bin_edges = np.histogram(hist_array, bins=DoD_bin_edges, range=hist_range)
            dist_array_sco_partial_class1_matrix[t,:] = hist/np.nansum(hist)
            
            hist_array = np.copy(dist_array_sco_partial_class2)
            hist_array = hist_array[~np.isnan(hist_array)] # Trim 0 and np.nan
            hist, bin_edges = np.histogram(hist_array, bins=DoD_bin_edges, range=hist_range)
            dist_array_sco_partial_class2_matrix[t,:] = hist/np.nansum(hist)
            
            hist_array = np.copy(dist_array_sco_partial_class3)
            hist_array = hist_array[~np.isnan(hist_array)] # Trim 0 and np.nan
            hist, bin_edges = np.histogram(hist_array, bins=DoD_bin_edges, range=hist_range)
            dist_array_sco_partial_class3_matrix[t,:] = hist/np.nansum(hist)
            
            hist_array = np.copy(dist_array_sco_partial_class4)
            hist_array = hist_array[~np.isnan(hist_array)] # Trim 0 and np.nan
            hist, bin_edges = np.histogram(hist_array, bins=DoD_bin_edges, range=hist_range)
            dist_array_sco_partial_class4_matrix[t,:] = hist/np.nansum(hist)
            
            hist_array = np.copy(dist_array_sco_partial_class5)
            hist_array = hist_array[~np.isnan(hist_array)] # Trim 0 and np.nan
            hist, bin_edges = np.histogram(hist_array, bins=DoD_bin_edges, range=hist_range)
            dist_array_sco_partial_class5_matrix[t,:] = hist/np.nansum(hist)
            
            hist_array = np.copy(dist_array_dep_partial_class6)
            hist_array = hist_array[~np.isnan(hist_array)] # Trim 0 and np.nan
            hist, bin_edges = np.histogram(hist_array, bins=DoD_bin_edges, range=hist_range)
            dist_array_dep_partial_class6_matrix[t,:] = hist/np.nansum(hist)
            
            hist_array = np.copy(dist_array_dep_partial_class7)
            hist_array = hist_array[~np.isnan(hist_array)] # Trim 0 and np.nan
            hist, bin_edges = np.histogram(hist_array, bins=DoD_bin_edges, range=hist_range)
            dist_array_dep_partial_class7_matrix[t,:] = hist/np.nansum(hist)
                
        # OVERALL DISTRIBUTION
        overall_dist_matrix = np.zeros((num_bins,num_bins))
        
        hist_array = np.copy(dist_array_sco_class1)
        hist_array = hist_array[~np.isnan(hist_array)] # Trim 0 and np.nan
        hist, bin_edges = np.histogram(hist_array, bins=bin_edges, range=hist_range)
        hist = hist/np.nansum(hist)
        np.savetxt(os.path.join(output_dir, 'class1_distribution_DEM2DoD.txt'), np.round(hist, decimals=2))
        overall_dist_matrix[0,:] = hist
        
        hist_array = np.copy(dist_array_sco_class2)
        hist_array = hist_array[~np.isnan(hist_array)] # Trim 0 and np.nan
        hist, bin_edges = np.histogram(hist_array, bins=bin_edges, range=hist_range)
        hist = hist/np.nansum(hist)
        np.savetxt(os.path.join(output_dir, 'class2_distribution_DEM2DoD.txt'), np.round(hist, decimals=2))
        overall_dist_matrix[1,:] = hist
        
        hist_array = np.copy(dist_array_sco_class3)
        hist_array = hist_array[~np.isnan(hist_array)] # Trim 0 and np.nan
        hist, bin_edges = np.histogram(hist_array, bins=bin_edges, range=hist_range)
        hist = hist/np.nansum(hist)
        np.savetxt(os.path.join(output_dir, 'class3_distribution_DEM2DoD.txt'), np.round(hist, decimals=2))
        overall_dist_matrix[2,:] = hist
        
        hist_array = np.copy(dist_array_sco_class4)
        hist_array = hist_array[~np.isnan(hist_array)] # Trim 0 and np.nan
        hist, bin_edges = np.histogram(hist_array, bins=bin_edges, range=hist_range)
        hist = hist/np.nansum(hist)
        np.savetxt(os.path.join(output_dir, 'class4_distribution_DEM2DoD.txt'), np.round(hist, decimals=2))
        overall_dist_matrix[3,:] = hist
        
        hist_array = np.copy(dist_array_sco_class5)
        hist_array = hist_array[~np.isnan(hist_array)] # Trim 0 and np.nan
        hist, bin_edges = np.histogram(hist_array, bins=bin_edges, range=hist_range)
        hist = hist/np.nansum(hist)
        np.savetxt(os.path.join(output_dir, 'class5_distribution_DEM2DoD.txt'), np.round(hist, decimals=2))
        overall_dist_matrix[4,:] = hist
        
        hist_array = np.copy(dist_array_dep_class6)
        hist_array = hist_array[~np.isnan(hist_array)] # Trim 0 and np.nan
        hist, bin_edges = np.histogram(hist_array, bins=bin_edges, range=hist_range)
        hist = hist/np.nansum(hist)
        np.savetxt(os.path.join(output_dir, 'class7_distribution_DEM2DoD.txt'), np.round(hist, decimals=2))
        overall_dist_matrix[5,:] = hist
        overall_dist_matrix
        
        hist_array = np.copy(dist_array_dep_class7)
        hist_array = hist_array[~np.isnan(hist_array)] # Trim 0 and np.nan
        hist, bin_edges = np.histogram(hist_array, bins=bin_edges, range=hist_range)
        hist = hist/np.nansum(hist)
        np.savetxt(os.path.join(output_dir, 'class7_distribution_DEM2DoD.txt'), np.round(hist, decimals=2))
        overall_dist_matrix[6,:] = hist
        
        np.savetxt(os.path.join(output_dir, run + '_overall_distribution_matrix_DEM2DoD.txt'), np.round(overall_dist_matrix, decimals=6))
        
        
    #%%
        if 'single_class_distribution' in plot_mode:    
            for i in range(overall_dist_matrix.shape[0]-1):
                
                # Define the data
                data = overall_dist_matrix[i,:]
                
                fig, ax = plt.subplots(figsize=(8, 6))
                
                # Plot the bars
                ax.bar(range(len(data)), data)
                
                
                # Set x axes ticks
                plt.xticks(range(len(data)), DoD_class_labels, rotation=45)  # Rotate x-axis labels for readability
        
                # Add values above each bar
                
                for x, y in zip(range(len(data)), data):
                    if np.logical_not(np.isnan(y)):
                        plt.text(x, y, str(np.round(y, decimals=3)), ha='center', va='bottom')
                 
                # Set the y limit
                # plt.ylim(0,0.50)
                plt.ylim(0,1.0)
                
                # Add labels and title to the plot
                plt.xlabel('DoD class')
                plt.ylabel('Y Values')
                plt.title(run + ' - DEM Class interval: ' + DEM_class_labels[i] + ' - Jump: ' + str(jump))
                # plt.xticks(bin_edges)
                # Save the figure to a file (e.g., 'scatter_plot.png')
                # plt.savefig(os.path.join(output_dir, run + '_Jump:' + str(jump) +  '_' + str(i) + '_overall_dest_probability_DEM2DoD.pdf'), dpi=300)
                
                
                # Show the plot
                plt.show()
            
    #%%
            
        if 'overall_class_distribution_stacked' in plot_mode:
            # GET DATA
            overall_dist_matrix = np.loadtxt(os.path.join(output_dir, run + '_overall_distribution_matrix_DEM2DoD.txt'))  
                    
            # sco = np.nansum(overall_dist_matrix[:,:5], axis=1)
            # null = np.nansum(overall_dist_matrix[:,5:6], axis=1)
            # dep = np.nansum(overall_dist_matrix[:,6:], axis=1)
            
            # Your data
            data1 = overall_dist_matrix[:,0]
            data2 = overall_dist_matrix[:,1]
            data3 = overall_dist_matrix[:,2]
            data4 = overall_dist_matrix[:,3]
            data5 = overall_dist_matrix[:,4]
            data6 = overall_dist_matrix[:,5]
            data7 = overall_dist_matrix[:,6]
            
            
            # TODO save this report that is useful to fill the Analisi.destinazioni_v1.ods
            # report = np.vstack((data1,data2,data3)).T
            # np.savetxt(os.path.join(output_dir, run + '_report_destination_distribution.txt'), report, delimiter=',', fmt='"%.4f"')
            
            
            # Bar width and alpha
            bar_width = 0.9
            alpha = 0.6
            
            # colors = ['#440154', '#443983', '#31688e', '#21918c', '#35b779', '#90d743', '#fde725'] # Viridis
            colors = ['#b40426', '#e7745b', '#f7b89c', '#dcdddd', '#aac7fd', '#6f92f3', '#3b4cc0'] # Coolwarm
            # Create a bar plot with stacked values and different colors
            bars1 = plt.bar(range(len(data1)), data1, width=bar_width, alpha=alpha, label=DoD_class_labels[0], color=colors[0])
            bars2 = plt.bar(range(len(data2)), data2, width=bar_width, alpha=alpha, bottom=data1, label=DoD_class_labels[1], color=colors[1])
            bars3 = plt.bar(range(len(data3)), data3, width=bar_width, alpha=alpha, bottom=data1 + data2, label=DoD_class_labels[2], color=colors[2])
            bars4 = plt.bar(range(len(data4)), data4, width=bar_width, alpha=alpha, bottom=data1 + data2 + data3, label=DoD_class_labels[3], color=colors[3])
            bars5 = plt.bar(range(len(data5)), data5, width=bar_width, alpha=alpha, bottom=data1 + data2 + data3 + data4, label=DoD_class_labels[4], color=colors[4])
            bars6 = plt.bar(range(len(data6)), data6, width=bar_width, alpha=alpha, bottom=data1 + data2 + data3 + data4 + data5, label=DoD_class_labels[5], color=colors[5])
            bars7 = plt.bar(range(len(data7)), data7, width=bar_width, alpha=alpha, bottom=data1 + data2 + data3 + data4 + data5 + data6, label=DoD_class_labels[6], color=colors[6])
            
            
            # Set x-axis labels
            plt.xticks(range(len(data1)), DEM_class_labels, rotation=45)  # Rotate x-axis labels for readability
        
            # Set a title
            plt.title('Stacked Bar Chart - ' + run + ' - Jump: ' + str(jump))
            
            
            
            # Label the axes
            plt.xlabel('DEM values')
            plt.ylabel('DoD values')
        
            # Add a legend
            legend = plt.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0), framealpha=0.5)
            
            # Customize legend alpha (transparency)
            legend.get_frame().set_alpha(0.5)
        
            # Add data labels with values centered within the bars
            for bar1, bar2, bar3, bar4, bar5, bar6, bar7 in zip(bars1, bars2, bars3, bars4, bars5, bars6, bars7):
                
                h1 = bar1.get_height()
                h2 = bar2.get_height()
                h3 = bar3.get_height()
                h4 = bar4.get_height()
                h5 = bar5.get_height()
                h6 = bar6.get_height()
                h7 = bar7.get_height()
                
                
                plt.text(bar1.get_x() + bar1.get_width() / 2, h1 / 2, str(np.round(h1, decimals=3)), ha='center', va='center', color='black')
                plt.text(bar2.get_x() + bar2.get_width() / 2, h1 + h2 / 2, str(np.round(h2, decimals=3)), ha='center', va='center', color='black')
                plt.text(bar3.get_x() + bar3.get_width() / 2, h1 + h2 + h3 / 2, str(np.round(h3, decimals=3)), ha='center', va='center', color='black')
                plt.text(bar4.get_x() + bar1.get_width() / 2, h1 + h2 + +h3 + h4 / 2, str(np.round(h4, decimals=3)), ha='center', va='center', color='black')
                plt.text(bar5.get_x() + bar2.get_width() / 2, h1 + h2 + h3 + h4 + h5 / 2, str(np.round(h5, decimals=3)), ha='center', va='center', color='black')
                plt.text(bar6.get_x() + bar3.get_width() / 2, h1 + h2 + h3 + h4 + h5 + h6 / 2, str(np.round(h6, decimals=3)), ha='center', va='center', color='black')
                plt.text(bar7.get_x() + bar3.get_width() / 2, h1 + h2 + h3 + h4 + h5 + h6 + h7 / 2, str(np.round(h7, decimals=3)), ha='center', va='center', color='black')
                
            plot_path = os.path.join(output_dir, run + '_Jump' + str(jump) + '_overall_distribution_DEM2DoD.pdf')
            plt.savefig(plot_path, dpi=300)
            # pdf_overall_dist_path = np.append(pdf_overall_dist_path, plot_path)
            # Show the plot
            plt.show()
        
        
            #%%        
        if 'overall_class_distribution_bars' in plot_mode:
            # Your data
            data1 = overall_dist_matrix[:,0]
            data2 = overall_dist_matrix[:,1]
            data3 = overall_dist_matrix[:,2]
            data4 = overall_dist_matrix[:,3]
            data5 = overall_dist_matrix[:,4]
            data6 = overall_dist_matrix[:,5]
            data7 = overall_dist_matrix[:,6]
            
            # Bar width and alpha
            bar_width = 0.6
            alpha = 0.6
            
            classes = DEM_class_labels
            class_distance = 1
            index = np.arange(len(classes)) * (bar_width * 7 + class_distance)
            
            # Create the bar chart
            plt.bar(index - 3*bar_width, data1, width=bar_width, label=DoD_class_labels[0], color=colors[0])
            plt.bar(index - 2*bar_width, data2, width=bar_width, label=DoD_class_labels[1], color=colors[1])
            plt.bar(index - bar_width, data3, width=bar_width, label=DoD_class_labels[2], color=colors[2])
            plt.bar(index, data4, width=bar_width, label=DoD_class_labels[3], color=colors[3])
            plt.bar(index + bar_width, data5, width=bar_width, label=DoD_class_labels[4], color=colors[4])
            plt.bar(index + 2*bar_width, data6, width=bar_width, label=DoD_class_labels[5], color=colors[5])
            plt.bar(index + 3*bar_width, data7, width=bar_width, label=DoD_class_labels[6], color=colors[6])
            
            # Configure the chart
            plt.xlabel('DEM Classes')
            plt.ylabel('DoD values Frequency')
            plt.title(run + ' - Jump: ' + str(jump))
            # Set x-axis labels
            plt.xticks(index, DEM_class_labels, rotation=45)  # Rotate x-axis labels for readability
            
            # Set the y limit
            # plt.ylim(0,0.50)
            plt.ylim(0,1.0)
            
            # Add a legend
            plt.legend()
            
            # Save image and report
            plot_path = os.path.join(output_dir, run + '_Jump' + str(jump) + '_overall_disstribution_sep_DEM2DoD.pdf')
            plt.savefig(plot_path, dpi=300)
            
            # Show the plot
            plt.show()
            
        
    #%%
        
        # SAVE THE MATRIX WITH THE 9 DISTRIBUTION, ONE FOR EACH TIMESTEP
        np.savetxt(os.path.join(output_dir, 'DEM2DoD_partial_class1_distribution.txt'), np.round(dist_array_sco_partial_class1_matrix, decimals=2), delimiter=',')
        np.savetxt(os.path.join(output_dir, 'DEM2DoD_partial_class2_distribution.txt'), np.round(dist_array_sco_partial_class2_matrix, decimals=2), delimiter=',')
        np.savetxt(os.path.join(output_dir, 'DEM2DoD_partial_class3_distribution.txt'), np.round(dist_array_sco_partial_class3_matrix, decimals=2), delimiter=',')
        np.savetxt(os.path.join(output_dir, 'DEM2DoD_partial_class4_distribution.txt'), np.round(dist_array_sco_partial_class4_matrix, decimals=2), delimiter=',')
        np.savetxt(os.path.join(output_dir, 'DEM2DoD_partial_class5_distribution.txt'), np.round(dist_array_sco_partial_class5_matrix, decimals=2), delimiter=',')
        np.savetxt(os.path.join(output_dir, 'DEM2DoD_partial_class6_distribution.txt'), np.round(dist_array_dep_partial_class6_matrix, decimals=2), delimiter=',')
        np.savetxt(os.path.join(output_dir, 'DEM2DoD_partial_class7_distribution.txt'), np.round(dist_array_dep_partial_class7_matrix, decimals=2), delimiter=',')
    
        
