#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 14:56:12 2023

@author: erri
"""


import numpy as np
import os
import matplotlib.pyplot as plt




# runs = ['q07_1', 'q10_2', 'q15_2', 'q20_2']

runs = ['q20_2']

delta = 1

num_bins = 120
hist_range = (-40, 40)  # Range of values to include in the histogram

for run in runs:
    home_dir = os.getcwd()
    report_dir = os.path.join(home_dir, 'output', run)
    plot_out_dir = os.path.join(home_dir, 'plot')
    
    data = np.load(os.path.join(home_dir,'DoDs', 'DoDs_stack', 'DoD_stack_'+run+'.npy'))
    
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
    
    dist_array_sco_partial_class1 = np.zeros((9,10000))
    dist_array_sco_partial_class2 = np.zeros((9,10000))
    dist_array_sco_partial_class3 = np.zeros((9,10000))
    dist_array_sco_partial_class4 = np.zeros((9,10000))
    dist_array_sco_partial_class5 = np.zeros((9,10000))
    dist_array_dep_partial_class6 = np.zeros((9,10000))
    dist_array_dep_partial_class7 = np.zeros((9,10000))
    dist_array_dep_partial_class8 = np.zeros((9,10000))
    dist_array_dep_partial_class9 = np.zeros((9,10000))
    dist_array_dep_partial_class10 = np.zeros((9,10000))
    

    
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
    matrix = data_dep
    for t in range(data.shape[0]-1):
        for x in range(data.shape[1]):
            for y in range(data.shape[2]):
                # if -40<=matrix[t,x,y]<-8:
                #     dist_array_dep_class1 = np.append(dist_array_dep_class1, data[t+1,x,y])
                #     dist_array_dep_partial_class1 = np.append(dist_array_dep_partial_class1, data[t+1,x,y])
                # if -8<=matrix[t,x,y]<-5.7:
                #     dist_array_dep_class2 = np.append(dist_array_dep_class2, data[t+1,x,y])
                #     dist_array_dep_partial_class2 = np.append(dist_array_dep_partial_class2, data[t+1,x,y])
                # if -5.7<=matrix[t,x,y]<-4.3:
                #     dist_array_dep_class3 = np.append(dist_array_dep_class3, data[t+1,x,y])
                #     dist_array_dep_partial_class3 = np.append(dist_array_dep_partial_class3, data[t+1,x,y])
                # if -4.3<=matrix[t,x,y]<-3.3:
                #     dist_array_dep_class4 = np.append(dist_array_dep_class4, data[t+1,x,y])
                #     dist_array_dep_partial_class4 = np.append(dist_array_dep_partial_class4, data[t+1,x,y])
                # if -3.3<=matrix[t,x,y]<0:
                #     dist_array_dep_class5 = np.append(dist_array_dep_class5, data[t+1,x,y])
                #     dist_array_dep_partial_class5 = np.append(dist_array_dep_partial_class5, data[t+1,x,y])
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
        
        hist, bin_edges = np.histogram(dist_array_dep_partial_class6, bins=num_bins, range=hist_range)
        dist_array_dep_partial_class6[t,:] = hist/np.nansum(hist)
        
        hist, bin_edges = np.histogram(dist_array_dep_partial_class7, bins=num_bins, range=hist_range)
        dist_array_dep_partial_class7[t,:] = hist/np.nansum(hist)
        
        hist, bin_edges = np.histogram(dist_array_dep_partial_class8, bins=num_bins, range=hist_range)
        dist_array_dep_partial_class8[t,:] = hist/np.nansum(hist)
        
        hist, bin_edges = np.histogram(dist_array_dep_partial_class9, bins=num_bins, range=hist_range)
        dist_array_dep_partial_class9[t,:] = hist/np.nansum(hist)
        
        hist, bin_edges = np.histogram(dist_array_dep_partial_class10, bins=num_bins, range=hist_range)
        dist_array_dep_partial_class10[t,:] = hist/np.nansum(hist)
        
        

        
    matrix = data_sco
    for t in range(data.shape[0]-1):
        
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

                    
    np.savetxt(os.path.join(report_dir, 'dist_step_by_step', 'partial_class1_distribution_sco_step'+str(t)+'.txt'), np.round(dist_array_sco_partial_class1, decimals=2))
    np.savetxt(os.path.join(report_dir, 'dist_step_by_step', 'partial_class2_distribution_sco_step'+str(t)+'.txt'), np.round(dist_array_sco_partial_class2, decimals=2))
    np.savetxt(os.path.join(report_dir, 'dist_step_by_step', 'partial_class3_distribution_sco_step'+str(t)+'.txt'), np.round(dist_array_sco_partial_class3, decimals=2))
    np.savetxt(os.path.join(report_dir, 'dist_step_by_step', 'partial_class4_distribution_sco_step'+str(t)+'.txt'), np.round(dist_array_sco_partial_class4, decimals=2))
    np.savetxt(os.path.join(report_dir, 'dist_step_by_step', 'partial_class5_distribution_sco_step'+str(t)+'.txt'), np.round(dist_array_sco_partial_class5, decimals=2))

    np.savetxt(os.path.join(report_dir, 'dist_step_by_step', 'partial_class6_distribution_dep_step'+str(t)+'.txt'), np.round(dist_array_dep_partial_class6, decimals=2))
    np.savetxt(os.path.join(report_dir, 'dist_step_by_step', 'partial_class7_distribution_dep_step'+str(t)+'.txt'), np.round(dist_array_dep_partial_class7, decimals=2))
    np.savetxt(os.path.join(report_dir, 'dist_step_by_step', 'partial_class8_distribution_dep_step'+str(t)+'.txt'), np.round(dist_array_dep_partial_class8, decimals=2))
    np.savetxt(os.path.join(report_dir, 'dist_step_by_step', 'partial_class9_distribution_dep_step'+str(t)+'.txt'), np.round(dist_array_dep_partial_class9, decimals=2))
    np.savetxt(os.path.join(report_dir, 'dist_step_by_step', 'partial_class10_distribution_dep_step'+str(t)+'.txt'), np.round(dist_array_dep_partial_class10, decimals=2))


# np.savetxt(os.path.join(report_dir, 'dist_step_by_step', 'class1_distribution_dep.txt'), dist_array_dep_class1)
# np.savetxt(os.path.join(report_dir, 'dist_step_by_step', 'class2_distribution_dep.txt'), dist_array_dep_class2)
# np.savetxt(os.path.join(report_dir, 'dist_step_by_step', 'class3_distribution_dep.txt'), dist_array_dep_class3)
# np.savetxt(os.path.join(report_dir, 'dist_step_by_step', 'class4_distribution_dep.txt'), dist_array_dep_class4)
# np.savetxt(os.path.join(report_dir, 'dist_step_by_step', 'class5_distribution_dep.txt'), dist_array_dep_class5)
np.savetxt(os.path.join(report_dir, 'dist_step_by_step', 'class6_distribution_dep.txt'), np.round(dist_array_dep_class6, decimals=2))
np.savetxt(os.path.join(report_dir, 'dist_step_by_step', 'class7_distribution_dep.txt'), np.round(dist_array_dep_class7, decimals=2))
np.savetxt(os.path.join(report_dir, 'dist_step_by_step', 'class8_distribution_dep.txt'), np.round(dist_array_dep_class8, decimals=2))
np.savetxt(os.path.join(report_dir, 'dist_step_by_step', 'class9_distribution_dep.txt'), np.round(dist_array_dep_class9, decimals=2))
np.savetxt(os.path.join(report_dir, 'dist_step_by_step', 'class10_distribution_dep.txt'), np.round(dist_array_dep_class10, decimals=2))

np.savetxt(os.path.join(report_dir, 'dist_step_by_step', 'class1_distribution_sco.txt'), np.round(dist_array_sco_class1, decimals=2))
np.savetxt(os.path.join(report_dir, 'dist_step_by_step', 'class2_distribution_sco.txt'), np.round(dist_array_sco_class2, decimals=2))
np.savetxt(os.path.join(report_dir, 'dist_step_by_step', 'class3_distribution_sco.txt'), np.round(dist_array_sco_class3, decimals=2))
np.savetxt(os.path.join(report_dir, 'dist_step_by_step', 'class4_distribution_sco.txt'), np.round(dist_array_sco_class4, decimals=2))
np.savetxt(os.path.join(report_dir, 'dist_step_by_step', 'class5_distribution_sco.txt'), np.round(dist_array_sco_class5, decimals=2))
# np.savetxt(os.path.join(report_dir, 'dist_step_by_step', 'class6_distribution_sco.txt'), dist_array_sco_class6)
# np.savetxt(os.path.join(report_dir, 'dist_step_by_step', 'class7_distribution_sco.txt'), dist_array_sco_class7)
# np.savetxt(os.path.join(report_dir, 'dist_step_by_step', 'class8_distribution_sco.txt'), dist_array_sco_class8)
# np.savetxt(os.path.join(report_dir, 'dist_step_by_step', 'class9_distribution_sco.txt'), dist_array_sco_class9)
# np.savetxt(os.path.join(report_dir, 'dist_step_by_step', 'class10_distribution_sco.txt'), dist_array_sco_class10)