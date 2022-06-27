#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 30 22:36:45 2022

@author: erri
"""

import numpy as np
import os
import math
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import mark_inset, inset_axes
from morph_quantities_func_v2 import morph_quantities

run = 'q20_2'
'''
Parameters:
    split_mode = 0: consider the entire array
    split_mode = 1: consider the upstream half of the array
    split_mode = 2: consider the downstream half of the array
'''
split_mode = 2

# DoD_list = ['DoD_s1-s0_filt_nozero_rst.txt'
#             , 'DoD_s2-s1_filt_nozero_rst.txt'
#             , 'DoD_s3-s2_filt_nozero_rst.txt'
#             , 'DoD_s4-s3_filt_nozero_rst.txt'
#             ]

home_dir = os.getcwd()
DoDs_dir = os.path.join(home_dir, 'DoDs', 'DoD_'+run)

# Domain_discretization
px_x = 0.05
px_y = 0.005
W_channel = 0.6/0.005 # Channel width in number of cell

window_mode = 3
windows_length_base = 12

W=windows_length_base

tot_vol_array = []
mean_array_tot = []
std_array_tot= []
x_data_tot=[]
n=0 #Count the number of iteration

DoD_delta = 1 # Step between surveys

# Loop to define DOD_name
DoDs_list = [] # List the file's name of the DoDs with step of delta_step
for f in sorted(os.listdir(DoDs_dir)):
    if f.endswith('_filt_nozero_rst.txt') and f.startswith('DoD_'):
        delta = eval(f[5]) - eval(f[8])
        if delta == DoD_delta:
            DoDs_list = np.append(DoDs_list, f)
        else:
            pass

# Loop to define the array dimension:
for DoD_name in DoDs_list:
    DoD_path = os.path.join(DoDs_dir, DoD_name)
    DoD = np.loadtxt(DoD_path, delimiter='\t')
    array = np.where(DoD==-999, np.nan, DoD)
    
# Slice array to perform statistics on half channel
if split_mode == 1:
    array = array[:,0:int(math.floor(array.shape[1]/2))]
elif split_mode == 2:
    array = array[:,int(math.ceil(array.shape[1]/2)):]
elif split_mode == 0:
    pass
        

# Create report matrix
# report_matrix = np.zeros(( len(DoD_list), np.size(morph_quantities(array))+2, len(window_boundary[:,0]) ))
report_matrix = np.zeros((len(DoDs_list), 16, 23))
if split_mode == 1 or split_mode == 2:
    report_matrix = np.zeros((len(DoDs_list), 16, 11))
cross_bri_matrix = np.zeros((len(DoDs_list), array.shape[1])) # q07_1, q15_2: 278, q10_2, q20_2: 279

for DoD_name in DoDs_list:
    n+=1
    print(DoD_name)
    DoD_path = os.path.join(DoDs_dir, DoD_name)
    DoD = np.loadtxt(DoD_path, delimiter='\t')
    array = np.where(DoD==-999, np.nan, DoD)
    
    # Slice array to perform statistics on half channel
    if split_mode == 1:
        array = array[:,0:int(math.floor(array.shape[1]/2))]
        print(array.shape)
    elif split_mode == 2:
        array = array[:,int(math.ceil(array.shape[1]/2)):]
        print(array.shape)
    elif split_mode == 0:
        pass
    
    if window_mode == 1:
        # With overlapping
        # Loop over all the available window dimension
        for w in range(1, int(math.floor(array.shape[1]/W))+1): # W*w is the dimension of every possible window   
            mean_array = []
            std_array= []
            x_data=[]
            window_boundary = np.array([0,0])
            for i in range(0, array.shape[1]+1):
                if i+w*W <= array.shape[1]:
                    window = array[:, i:W*w+i]
                    boundary = np.array([i,W*w+i])
                    window_boundary = np.vstack((window_boundary, boundary))
                    mean = np.nanmean(window)
                    std = np.nanstd(window)
                    mean_array = np.append(mean_array, mean)
                    std_array = np.append(std_array, std)
                    x_data=np.append(x_data, w)       
            mean_array_tot = np.append(mean_array_tot, np.nanmean(mean_array))
            std_array_tot= np.append(std_array_tot, np.nanstd(std_array)) #TODO check this
            x_data_tot=np.append(x_data_tot, np.nanmean(x_data))
            window_boundary = window_boundary[1:,:]
    
    if window_mode == 2:
        # Without overlapping
        # Loop over all the available window dimension
        for w in range(1, int(math.floor(array.shape[1]/W))+1): # W*w is the dimension of every possible window
            x_data=[]
            tot_vol_w_array = []
            sum_vol_w_array = []
            dep_vol_w_array = []
            sco_vol_w_array = []
            morph_act_area_w_array = []
            morph_act_area_dep_w_array = []
            morph_act_area_sco_w_array = []
            act_width_mean_w_array = []
            act_width_mean_dep_w_array = []
            act_width_mean_sco_w_array = []
            act_thickness_w_array = []
            act_thickness_dep_w_array = []
            act_thickness_sco_w_array = []
            window_boundary = np.array([0,0])
            for i in range(0, array.shape[1]+1): # For each window, loop over all available positions
                if W*w*(i+1) <= array.shape[1]:
                    window = array[:, W*w*i:W*w*(i+1)]
                    boundary = np.array([W*w*i,W*w*(i+1)])
                    window_boundary = np.vstack((window_boundary, boundary))
                    
                    # Calculate morphological quantities:
                    tot_vol, sum_vol, dep_vol, sco_vol, morph_act_area, morph_act_area_dep, morph_act_area_sco, act_width_mean, act_width_mean_dep, act_width_mean_sco, act_thickness, act_thickness_dep, act_thickness_sco = morph_quantities(window)
                    
                    # Append morphological quantities into array
                    x_data=np.append(x_data, w)
                    tot_vol_w_array = np.append(tot_vol_w_array, tot_vol)
                    sum_vol_w_array = np.append(sum_vol_w_array, sum_vol)
                    dep_vol_w_array = np.append(dep_vol_w_array, dep_vol)
                    sco_vol_w_array = np.append(sco_vol_w_array, sco_vol)
                    morph_act_area_w_array = np.append(morph_act_area_w_array, morph_act_area)
                    morph_act_area_dep_w_array = np.append(morph_act_area_dep_w_array, morph_act_area_dep)
                    morph_act_area_sco_w_array = np.append(morph_act_area_sco_w_array, morph_act_area_sco)
                    act_width_mean_w_array = np.append(act_width_mean_w_array, act_width_mean)
                    act_width_mean_dep_w_array = np.append(act_width_mean_dep_w_array, act_width_mean_dep)
                    act_width_mean_sco_w_array = np.append(act_width_mean_sco_w_array, act_width_mean_sco)
                    act_thickness_w_array = np.append(act_thickness_w_array, act_thickness)
                    act_thickness_dep_w_array = np.append(act_thickness_dep_w_array, act_thickness_dep)
                    act_thickness_sco_w_array = np.append(act_thickness_sco_w_array, act_thickness_sco)
                    # print(act_thickness_sco_w_array)
        
            # if DoD_name == DoD_list[0]:
            if w == 1:
                tot_vol_array = tot_vol_w_array
                sum_vol_array = sum_vol_w_array
                dep_vol_array = dep_vol_w_array
                sco_vol_array = sco_vol_w_array
                morph_act_area_array = morph_act_area_w_array
                morph_act_area_dep_array = morph_act_area_dep_w_array
                morph_act_area_sco_array = morph_act_area_sco_w_array
                act_width_mean_array = act_width_mean_w_array
                act_width_mean_dep_array = act_width_mean_dep_w_array
                act_width_mean_sco_array = act_width_mean_sco_w_array
                act_thickness_array = act_thickness_w_array
                act_thickness_dep_array = act_thickness_dep_w_array
                act_thickness_sco_array = act_thickness_sco_w_array
            else:
                tot_vol_array = np.append(tot_vol_array, tot_vol_w_array)
                sum_vol_array = np.append(sum_vol_array, sum_vol_w_array)
                dep_vol_array = np.append(dep_vol_array, dep_vol_w_array)
                sco_vol_array = np.append(sco_vol_array, sco_vol_w_array)
                morph_act_area_array = np.append(morph_act_area_array, morph_act_area_w_array)
                morph_act_area_dep_array = np.append(morph_act_area_dep_array, morph_act_area_dep_w_array)
                morph_act_area_sco_array = np.append(morph_act_area_sco_array, morph_act_area_sco_w_array)
                act_width_mean_array = np.append(act_width_mean_array, act_width_mean_w_array)
                act_width_mean_dep_array = np.append(act_width_mean_dep_array, act_width_mean_dep_w_array)
                act_width_mean_sco_array = np.append(act_width_mean_sco_array, act_width_mean_sco_w_array)
                act_thickness_array = np.append(act_thickness_array, act_thickness_w_array)
                act_thickness_dep_array = np.append(act_thickness_dep_array, act_thickness_dep_w_array)
                act_thickness_sco_array = np.append(act_thickness_sco_array, act_thickness_sco_w_array)
                x_data_tot = np.append(x_data,x_data)        
                # x_data_tot=np.append(x_data_tot, np.nanmean(x_data))
                window_boundary = window_boundary[1:,:]
    
    if window_mode == 3:
        # Increasing window dimension keeping still the upstream cross section
        
        x_data=[]
        tot_vol_w_array = []
        sum_vol_w_array = []
        dep_vol_w_array = []
        sco_vol_w_array = []
        morph_act_area_w_array = []
        morph_act_area_dep_w_array = []
        morph_act_area_sco_w_array = []
        act_width_mean_w_array = []
        act_width_mean_dep_w_array = []
        act_width_mean_sco_w_array = []
        act_thickness_w_array = []
        act_thickness_dep_w_array = []
        act_thickness_sco_w_array = []
        bri_w_array=[]
        bri_crosswise_array=[]
        window_boundary = np.array([0,0])
        
        for k in range(0, array.shape[1]):
            bri_crosswise=np.nanstd(array[:,k])
            bri_crosswise_array=np.append(bri_crosswise_array, bri_crosswise)
        cross_bri_matrix[n-1,:] = bri_crosswise_array
        
        for i in range(0, array.shape[1]+1):
            if W*(i+1) <= array.shape[1]:
                window = array[:, 0:W*(i+1)] # Slicing of the main array
                boundary = np.array([0,W*(i+1)]) # Window boundarys
                window_boundary = np.vstack((window_boundary, boundary)) # Stack window boundary in array

                # Calculate morphological quantities:
                tot_vol, sum_vol, dep_vol, sco_vol, morph_act_area, morph_act_area_dep, morph_act_area_sco, act_width_mean, act_width_mean_dep, act_width_mean_sco, act_thickness, act_thickness_dep, act_thickness_sco, bri = morph_quantities(window)
                
                # Append morphological quantities into array
                # Data were scaled with the window dimension W*(i+1)
                x_data=np.append(x_data, i)
                tot_vol_w_array = np.append(tot_vol_w_array, tot_vol/(i+1))
                sum_vol_w_array = np.append(sum_vol_w_array, sum_vol/(i+1))
                dep_vol_w_array = np.append(dep_vol_w_array, dep_vol/(i+1))
                sco_vol_w_array = np.append(sco_vol_w_array, sco_vol/(i+1))
                morph_act_area_w_array = np.append(morph_act_area_w_array, morph_act_area/(i+1))
                morph_act_area_dep_w_array = np.append(morph_act_area_dep_w_array, morph_act_area_dep/(i+1))
                morph_act_area_sco_w_array = np.append(morph_act_area_sco_w_array, morph_act_area_sco/(i+1))
                act_width_mean_w_array = np.append(act_width_mean_w_array, act_width_mean/W_channel)
                act_width_mean_dep_w_array = np.append(act_width_mean_dep_w_array, act_width_mean_dep/W_channel)
                act_width_mean_sco_w_array = np.append(act_width_mean_sco_w_array, act_width_mean_sco/W_channel)
                act_thickness_w_array = np.append(act_thickness_w_array, act_thickness)
                act_thickness_dep_w_array = np.append(act_thickness_dep_w_array, act_thickness_dep)
                act_thickness_sco_w_array = np.append(act_thickness_sco_w_array, act_thickness_sco)
                bri_w_array = np.append(bri_w_array, bri)
                
        window_boundary = window_boundary[1:,:] # Slicing window bundary array deliting the first entry used to initialization
        

        # Create morphological parameter matrix
        matrix = np.vstack((np.transpose(window_boundary),
                      tot_vol_w_array,sum_vol_w_array,dep_vol_w_array,sco_vol_w_array,
                      morph_act_area_w_array,morph_act_area_dep_w_array,morph_act_area_sco_w_array,
                      act_width_mean_w_array,act_width_mean_dep_w_array,act_width_mean_sco_w_array,
                      act_thickness_w_array,act_thickness_dep_w_array,act_thickness_sco_w_array,
                      bri_w_array))
    
    # Fill report matrix
    report_matrix[n-1,:,:] = matrix[:,:]

# fig1, axs = plt.subplots(1,1,dpi=200, sharex=True, tight_layout=True)
# axs.plot(np.linspace(0,array.shape[1]-1, array.shape[1])*px_x, cross_bri_matrix[n-1,:], linestyle='--', marker='^', color='green')
# axs.set_title(run)
# axs.set_xlabel('Longitudinal coordinate [m]')
# axs.set_ylabel('BRI')
# # plt.savefig(os.path.join(plot_dir, run +'_morphW_interp.png'), dpi=200)
# plt.show()

mean_matrix = np.mean(report_matrix[:,:,:], axis=0)
std_matrix = np.std(report_matrix[:,:,:], axis=0)

# Plot
n_data = np.linspace(0,len(report_matrix[:,0,0])-1,len(report_matrix[:,0,0])) # Linspace of the number of available DoD
c_data = np.linspace(0,1,len(report_matrix[:,0,0])) # c_data needs to be within 0 and 1
# p_data = np.linspace(2,len(report_matrix[0,:,0])-1, len(report_matrix[0,:,0])-2) # Linspace of the number of parameters
colors = plt.cm.viridis(c_data)
lables = ['Total volume', 'Sum volume', 'Deposition volume', 'Scour volume', 'Morphological active area',
          'Morphological active deposition area', 'Morphological active scour area', 'Total active width', 'Active deposition width', 'Active scour width',
          'Total active thickness', 'Active deposition thickness', 'Active scour thickness', 'Bed Relief Index [mm]']
for i in range(2,len(report_matrix[0,:,0])):
    label = lables[i-2]
    fig1, axs = plt.subplots(1,1,dpi=200, sharex=True, tight_layout=True, figsize=(8,6))
    for d, color in zip(n_data, colors):
        DoD_name = DoDs_list[int(d)]
        axs.plot(report_matrix[int(d),1,:]*px_x, report_matrix[int(d),i,:], '-o', c=color, label=DoD_name[:9])
    axs.set_title(run, fontsize=14)
    axs.set_xlabel('Window analysis length [m]', fontsize=12)
    axs.set_ylabel(label, fontsize=12)
    # plt.savefig(os.path.join(plot_dir, run +'_morphW_interp.png'), dpi=200)
    plt.legend(loc='best', fontsize=8)
    plt.show()
    
    
    fig1, axs = plt.subplots(1,1,dpi=200, sharex=True, tight_layout=True, figsize=(8,6))
    axs.errorbar(mean_matrix[1,:]*px_x, mean_matrix[i,:], std_matrix[i,:], linestyle='--', marker='^', color='darkcyan')
    plt.axhline(y=mean_matrix[i,-1]+std_matrix[i,-1], color='red', linestyle='--')
    plt.axhline(y=mean_matrix[i,-1]-std_matrix[i,-1], color='red', linestyle='--')
    plt.axvline(x=6, color='dimgrey', linestyle='--')
    axs.set_title(run, fontsize=14)
    axs.set_xlabel('Window analysis length [m]', fontsize=12)
    axs.set_ylabel(label, fontsize=12)
    # plt.savefig(os.path.join(plot_dir, run +'_morphW_interp.png'), dpi=200)
    plt.show()

    
# fig1, axs = plt.subplots(1,1,dpi=400, sharex=True, tight_layout=True, figsize=(8,6))
# ax_new = fig1.add_axes([0.2, 1.1, 0.4, 0.4])
# for d, color in zip(n_data, colors):
#     axs.plot(np.linspace(0,array.shape[1]-1, array.shape[1])*px_x, cross_bri_matrix[int(d),:], '-', c=color, label=DoD_name[:9])
    
#     plt.plot(np.linspace(0,array.shape[1]-1, array.shape[1])*px_x, cross_bri_matrix[int(d),:], '-', c=color, label=DoD_name[:9])
# axs.set_title(run, fontsize=14)
# axs.set_xlabel('Longitudinal coordinate [m]', fontsize=12)
# axs.set_ylabel('BRI', fontsize=12)
# # plt.savefig(os.path.join(plot_dir, run +'_morphW_interp.png'), dpi=200)
# plt.legend(loc='best', fontsize=8)
# plt.show()

# BRI plot
if split_mode == 0:
    fig1, axs = plt.subplots(1,1,dpi=400, sharex=True, tight_layout=True, figsize=(8,6))
    #Defines the size of the zoom window and the positioning
    axins = inset_axes(axs, 2, 4, loc = 1, bbox_to_anchor=(1.4, 1.1),
                       bbox_transform = axs.figure.transFigure)
    for d, color in zip(n_data, colors):
        axs.plot(np.linspace(0,array.shape[1]-1, array.shape[1])*px_x, cross_bri_matrix[int(d),:], '-', c=color, label=DoD_name[:9])
        plt.plot(np.linspace(0,array.shape[1]-1, array.shape[1])*px_x, cross_bri_matrix[int(d),:], '-', c=color, label=DoD_name[:9])
    
    # axins.scatter(x, y)
    x1, x2 = 12, 14
    y1, y2 = 0, 10 #Setting the limit of x and y direction to define which portion to #zoom
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    #Draw the lines from the portion to zoom and the zoom window
    mark_inset(axs, axins, loc1=1, loc2=3, fc="none", ec = "0.4")
    axs.set_title(run, fontsize=14)
    axs.set_xlabel('Longitudinal coordinate [m]', fontsize=12)
    axs.set_ylabel('BRI', fontsize=12)
    # plt.savefig(os.path.join(plot_dir, run +'_morphW_interp.png'), dpi=200)
    plt.legend(loc='best', fontsize=8)
    
    # ax_new = fig1.add_axes([0.2, 1.1, 0.4, 0.4])
    # plt.plot(np.linspace(0,array.shape[1]-1, array.shape[1])*px_x, cross_bri_matrix[int(d),:], '-', c=color)
    
    plt.show()
else:
    pass