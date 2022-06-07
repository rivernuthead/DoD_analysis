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
from morph_quantities_func_v2 import morph_quantities

run = 'q07_1'
DoD_list = ['DoD_s1-s0_filt_nozero_rst.txt'
            , 'DoD_s2-s1_filt_nozero_rst.txt'
            , 'DoD_s3-s2_filt_nozero_rst.txt'
            , 'DoD_s4-s3_filt_nozero_rst.txt'
            ]

home_dir = os.getcwd()
DoDs_dir = os.path.join(home_dir, 'DoDs')


window_mode = 3
windows_length_base = 12

W=windows_length_base

tot_vol_array = []
mean_array_tot = []
std_array_tot= []
x_data_tot=[]
n=0 #Count the number of iteration


# TODO creare un ciclo fittizio per ricavare i parametri chiave per inizializzare la matrice
# Create report matrix
# report_matrix = np.zeros(( len(DoD_list), np.size(morph_quantities(array))+2, len(window_boundary[:,0]) ))
report_matrix = np.zeros((4, 15, 23))

for DoD_name in DoD_list:
    n+=1
    print(DoD_name)
    DoD_path = os.path.join(DoDs_dir, 'DoD_' + run, DoD_name)
    DoD = np.loadtxt(DoD_path, delimiter='\t')
    array = np.where(DoD==-999, np.nan, DoD)
    
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
        window_boundary = np.array([0,0])
        
        # Create report matrix
        # A = np.zeros(( len(DoD_list), np.size(morph_quantities(array))+2, len(window_boundary[:,0]) ))
        
        
        for i in range(0, array.shape[1]+1):
            
            if W*(i+1) <= array.shape[1]:
                window = array[:, 0:W*(i+1)] # Slicing of the main array
                boundary = np.array([0,W*(i+1)]) # Window boundarys
                window_boundary = np.vstack((window_boundary, boundary)) # Stack window boundary in array

                # Calculate morphological quantities:
                tot_vol, sum_vol, dep_vol, sco_vol, morph_act_area, morph_act_area_dep, morph_act_area_sco, act_width_mean, act_width_mean_dep, act_width_mean_sco, act_thickness, act_thickness_dep, act_thickness_sco = morph_quantities(window)
                
                # Append morphological quantities into array
                # Data were scaled with the window dimension W*(i+1)
                x_data=np.append(x_data, i)
                tot_vol_w_array = np.append(tot_vol_w_array, tot_vol/(W*(i+1)))
                sum_vol_w_array = np.append(sum_vol_w_array, sum_vol/(W*(i+1)))
                dep_vol_w_array = np.append(dep_vol_w_array, dep_vol/(W*(i+1)))
                sco_vol_w_array = np.append(sco_vol_w_array, sco_vol/(W*(i+1)))
                morph_act_area_w_array = np.append(morph_act_area_w_array, morph_act_area/(W*(i+1)))
                morph_act_area_dep_w_array = np.append(morph_act_area_dep_w_array, morph_act_area_dep/(W*(i+1)))
                morph_act_area_sco_w_array = np.append(morph_act_area_sco_w_array, morph_act_area_sco/(W*(i+1)))
                act_width_mean_w_array = np.append(act_width_mean_w_array, act_width_mean/(W*(i+1)))
                act_width_mean_dep_w_array = np.append(act_width_mean_dep_w_array, act_width_mean_dep/(W*(i+1)))
                act_width_mean_sco_w_array = np.append(act_width_mean_sco_w_array, act_width_mean_sco/(W*(i+1)))
                act_thickness_w_array = np.append(act_thickness_w_array, act_thickness/(W*(i+1)))
                act_thickness_dep_w_array = np.append(act_thickness_dep_w_array, act_thickness_dep/(W*(i+1)))
                act_thickness_sco_w_array = np.append(act_thickness_sco_w_array, act_thickness_sco/(W*(i+1)))
                
        window_boundary = window_boundary[1:,:] # Slicing window bundary array deliting the first entry used to initialization
        

        # Create morphological parameter matrix
        matrix = np.vstack((np.transpose(window_boundary),
                      tot_vol_w_array,sum_vol_w_array,dep_vol_w_array,sco_vol_w_array,
                      morph_act_area_w_array,morph_act_area_dep_w_array,morph_act_area_sco_w_array,
                      act_width_mean_w_array,act_width_mean_dep_w_array,act_width_mean_sco_w_array,
                      act_thickness_w_array,act_thickness_dep_w_array,act_thickness_sco_w_array))
    
    # Fill report matrix
    report_matrix[n-1,:,:] = matrix[:,:]
    
            


mean_matrix = np.mean(report_matrix[:,:,:], axis=0)
std_matrix = np.std(report_matrix[:,:,:], axis=0)

# Plot
n_data = np.linspace(0,len(report_matrix[:,0,0])-1,len(report_matrix[:,0,0]))
c_data = np.linspace(0,1,len(report_matrix[:,0,0])) # c_data needs to be within 0 and 1
colors = plt.cm.viridis(c_data)
fig1, axs = plt.subplots(1,1,dpi=200, sharex=True, tight_layout=True)
for d, color in zip(n_data, colors):
    DoD_name = DoD_list[int(d)]
    print(DoD_name)
    axs.plot(report_matrix[int(d),1,:], report_matrix[int(d),4,:], 'o', c=color, label=DoD_name[:9])
axs.set_title(run)
# axs.set_xlabel('Window analysis length [m]')
# axs.set_ylabel('Active thickness [mm]')
# plt.savefig(os.path.join(plot_dir, run +'_morphW_interp.png'), dpi=200)
plt.legend(loc='best', fontsize=8)
plt.show()

fig1, axs = plt.subplots(1,1,dpi=200, sharex=True, tight_layout=True)
axs.errorbar(mean_matrix[1,:], mean_matrix[4,:], std_matrix[4,:], linestyle='--', marker='^', color='brown')
# axs.set_title(run)
# axs.set_xlabel('Window analysis length [m]')
# axs.set_ylabel('Deposition volumes V/(W*L*d50) [-]')
# plt.savefig(os.path.join(plot_dir, run +'_morphW_interp.png'), dpi=200)
plt.show()
