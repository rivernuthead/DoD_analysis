#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 25 16:02:58 2022

@author: erri
"""
import os
import numpy as np
import math
import matplotlib.pyplot as plt

# SINGLE RUN NAME
run = 'q15_2'
DoD_name = 'DoD_s1-s0_filt_nozero_rst.txt'
# Step between surveys
DoD_delta = 1

windows_mode = 3

'''
windows_mode:
    0 = fixed windows (all the channel)
    1 = expanding window
    2 = floating fixed windows (WxW, Wx2W, Wx3W, ...) without overlapping
    3 = floating fixed windows (WxW, Wx2W, Wx3W, ...) with overlapping
'''

plot_mode = 2
'''
plot_mode:
    1 = only summary plot
    2 = all single DoD plot
'''
# Parameters
# Survey pixel dimension
px_x = 50 # [mm]
px_y = 5 # [mm]
W = 0.6 # Width [m]
d50 = 0.001
NaN = -999

# setup working directory and DEM's name
home_dir = os.getcwd()
# Source DoDs folder
DoDs_folder = os.path.join(home_dir, 'DoDs', 'DoD_'+run)

DoDs_name_array = [] # List the file's name of the DoDs with step of delta_step
for f in sorted(os.listdir(DoDs_folder)):
    if f.endswith('_filt_nozero_rst.txt') and f.startswith('DoD_'):
        delta = eval(f[5]) - eval(f[8])
        if delta == DoD_delta:
            DoDs_name_array = np.append(DoDs_name_array, f)
        else:
            pass
        
# Initialize overall arrays
dep_vol_w_array_all = []
sco_vol_w_array_all = []

# Loop over the DoDs with step of delta_step
for f in DoDs_name_array:
    DoD_name = f
    print(f)
    
    DoD_path = os.path.join(DoDs_folder,DoD_name)
    DoD_filt_nozero = np.loadtxt(DoD_path, delimiter='\t')
    
    # DoD length
    DoD_length = DoD_filt_nozero.shape[1]*px_x/1000 # DoD length [m]
    dim_x = DoD_filt_nozero.shape[1]
    
    # Initialize array
    volumes_array=[] # Tot volume
    dep_array=[] # Deposition volume
    sco_array=[] # Scour volume
    sum_array=[] # Sum of scour and deposition volume
    morph_act_area_array=[] # Total active area array
    morph_act_area_array_dep=[] # Deposition active area array
    morph_act_area_array_sco=[] # Active active area array
    act_width_mean_array=[] # Total active width mean array
    act_width_mean_array_dep=[] # Deposition active width mean array
    act_width_mean_array_sco=[] # Scour active width mean array
    
    # Define total volume matrix, Deposition matrix and Scour matrix
    DoD_vol = np.where(np.isnan(DoD_filt_nozero), 0, DoD_filt_nozero) # Total volume matrix
    DoD_vol = np.where(DoD_vol==NaN, 0, DoD_vol)
    dep_DoD = (DoD_vol>0)*DoD_vol # DoD of only deposition data
    sco_DoD = (DoD_vol<0)*DoD_vol # DoD of only scour data
    
    tot_vol = np.sum(DoD_vol)*px_x*px_y/(W*DoD_length*d50*1e09) # Total volume as V/(L*W*d50) [-] considering negative sign for scour
    sum_vol = np.sum(np.abs(DoD_vol))*px_x*px_y/(W*DoD_length*d50*1e09) # Sum of scour and deposition volume as V/(L*W*d50) [-]
    dep_vol = np.sum(dep_DoD)*px_x*px_y/(W*DoD_length*d50*1e09) # Deposition volume as V/(L*W*d50) [-]
    sco_vol = np.sum(sco_DoD)*px_x*px_y/(W*DoD_length*d50*1e09) # Scour volume as V/(L*W*d50) [-]
    
    
    # #Print results:
    # print('Total volume V/(L*W*d50) [-]:', "{:.1f}".format(tot_vol))
    # print('Sum of deposition and scour volume V/(L*W*d50) [-]:', "{:.1f}".format(sum_vol))
    # print('Deposition volume V/(L*W*d50) [-]:', "{:.1f}".format(dep_vol))
    # print('Scour volume V/(L*W*d50) [-]:', "{:.1f}".format(sco_vol))
    
    # Append values to output data array
    volumes_array = np.append(volumes_array, tot_vol)
    dep_array = np.append(dep_array, dep_vol)
    sco_array = np.append(sco_array, sco_vol)
    sum_array = np.append(sum_array, sum_vol)
    
    
    ###################################################################
    # Active_pixel analysis
    ###################################################################
    
    act_px_matrix = np.where(DoD_vol!=0, 1, 0) # Active pixel matrix, both scour and deposition
    act_px_matrix_dep = np.where(dep_DoD != 0, 1, 0) # Active deposition matrix 
    act_px_matrix_sco = np.where(sco_DoD != 0, 1, 0) # Active scour matrix
    
    morph_act_area = np.count_nonzero(act_px_matrix)*px_x*px_y # Active area both in terms of scour and deposition [mm²]
    morph_act_area_dep = np.count_nonzero(act_px_matrix_dep)*px_x*px_y # Active deposition area [mm²]
    morph_act_area_sco = np.count_nonzero(act_px_matrix_sco)*px_x*px_y # Active scour area [mm²]
    
    morph_act_area_array = np.append(morph_act_area_array, morph_act_area) # For each DoD, append total active area data
    morph_act_area_array_dep = np.append(morph_act_area_array_dep, morph_act_area_dep) # For each DoD, append deposition active area data
    morph_act_area_array_sco = np.append(morph_act_area_array_sco, morph_act_area_sco) # For each DoD, append scour active area data
    
    act_width_mean = (morph_act_area/(DoD_length*1000))/(W*1000) # Total mean active width [%] - Wact/W
    act_width_mean_dep = (morph_act_area_dep/(DoD_length*1000))/(W*1000) # Deposition mean active width [%] - Wact/W
    act_width_mean_sco = (morph_act_area_sco/(DoD_length*1000))/(W*1000) # Scour mean active width [%] - Wact/W
    
    act_width_mean_array = np.append(act_width_mean_array, act_width_mean) # For each DoD append total active width values
    act_width_mean_array_dep = np.append(act_width_mean_array_dep, act_width_mean_dep) # For each DoD append deposition active width values
    act_width_mean_array_sco = np.append(act_width_mean_array_sco, act_width_mean_sco) # For each DoD append scour active width values
    
    act_width_array = np.array([np.nansum(act_px_matrix, axis=0)])*px_y/1000/W # Array of the crosswise morphological total active width [Wact/W]
    act_width_array_dep = np.array([np.nansum(act_px_matrix_dep, axis=0)])*px_y/1000/W # Array of the crosswise morphological deposition active width [Wact/W]
    act_width_array_sco = np.array([np.nansum(act_px_matrix_sco, axis=0)])*px_y/1000/W # Array of the crosswise morphological scour active width [Wact/W]
    
    # Calculate active thickness for total volumes. deposition volumes and scour volumes
    act_thickness = (np.sum(np.abs(DoD_vol))*px_x*px_y)/morph_act_area # Total active thickness (abs(V_sco) + V_dep)/act_area [mm]
    act_thickness_dep = (np.sum(np.abs(dep_DoD))*px_x*px_y)/morph_act_area_dep # Deposition active thickness (abs(V_sco) + V_dep)/act_area [mm]
    act_thickness_sco = (np.sum(np.abs(sco_DoD))*px_x*px_y)/morph_act_area_sco # Scour active thickness (abs(V_sco) + V_dep)/act_area [mm]
    
    # print('Active thickness [mm]:', act_thickness)
    # print('Morphological active area: ', "{:.1f}".format(morph_act_area), '[mm²]')
    # print('Morphological active width (mean):', "{:.3f}".format(act_width_mean), '%')
    # print()
    # print()
    
    
    # Initialize array
    tot_vol_w_array = []
    sum_vol_w_array = []
    dep_vol_w_array = []
    sco_vol_w_array =[]
    morph_act_area_w_array = []
    morph_act_area_dep_w_array = []
    morph_act_area_sco_w_array = []
    act_width_mean_w_array = []
    act_width_mean_dep_w_array = []
    act_width_mean_sco_w_array = []
    act_thickness_w_array = []
    act_thickness_dep_w_array = []
    act_thickness_sco_w_array = []
    
    
    ###################################################################
    # MOVING WINDOWS ANALYSIS
    ###################################################################
    
    # TODO Go on with this section
    if windows_mode == 1:
    
        # Define x_data for plots
        x_data = np.linspace(W,dim_x,math.floor(DoD_length/W))*px_x/1e03
        for n in range(1,math.floor(DoD_length/W)+1):
            w_cols = n*round(W/(px_x/1000)) # Window analysis length in number of columns
            w_len = round(n*W,1) # Window analysis lenght im meter [m]
            
            # Define total volume matrix, Deposition matrix and Scour matrix
            DoD_vol_w = DoD_vol[:,0:w_cols] # Total volume matrix
            dep_DoD_w = dep_DoD[:,0:w_cols] # DoD of only deposition data
            sco_DoD_w = sco_DoD[:,0:w_cols] # DoD of only scour data
            
            # Define active pixel matrix
            act_px_matrix_w = act_px_matrix[:,0:w_cols] # Active pixel matrix, both scour and deposition
            act_px_matrix_dep_w = act_px_matrix_dep[:,0:w_cols] # Active deposition matrix 
            act_px_matrix_sco_w = act_px_matrix_sco[:,0:w_cols] # Active scour matrix
            
            # Calculate principal quantities:
            # Volumes
            tot_vol_w = np.sum(DoD_vol_w)*px_x*px_y/(W*w_len*d50*1e09)# Total volume as V/(L*W*d50) [-] considering negative sign for scour
            sum_vol_w = np.sum(np.abs(DoD_vol_w))*px_x*px_y/(W*w_len*d50*1e09) # Sum of scour and deposition volume as V/(L*W*d50) [-]
            dep_vol_w = np.sum(dep_DoD_w)*px_x*px_y/(W*w_len*d50*1e09) # Deposition volume as V/(L*W*d50) [-]
            sco_vol_w = np.sum(sco_DoD_w)*px_x*px_y/(W*w_len*d50*1e09) # Scour volume as V/(L*W*d50) [-]
            
            # Areas:
            morph_act_area_w = np.count_nonzero(act_px_matrix_w)*px_x*px_y/(W*w_len*1e06) # Active area both in terms of scour and deposition as A/(W*L) [-]
            morph_act_area_dep_w = np.count_nonzero(act_px_matrix_dep_w)*px_x*px_y/(W*w_len*1e06) # Active deposition area as A/(W*L) [-]
            morph_act_area_sco_w = np.count_nonzero(act_px_matrix_sco_w)*px_x*px_y/(W*w_len*1e06) # Active scour area as A/(W*L) [-]
            
            # Widths:
            act_width_mean_w = np.count_nonzero(act_px_matrix_w)*px_x*px_y/(W*w_len*1e06) # Total mean active width [%] - Wact/W
            act_width_mean_dep_w = np.count_nonzero(act_px_matrix_dep_w)*px_x*px_y/(W*w_len*1e06) # Deposition mean active width [%] - Wact/W
            act_width_mean_sco_w = np.count_nonzero(act_px_matrix_sco_w)*px_x*px_y/(W*w_len*1e06) # Scour mean active width [%] - Wact/W
            
            # Thicknesses:
            act_thickness_w = sum_vol_w/morph_act_area_w*(d50*1e03) # Total active thickness (abs(V_sco) + V_dep)/act_area [mm]
            act_thickness_dep_w = dep_vol_w/morph_act_area_dep_w*(d50*1e03) # Deposition active thickness V_dep/act_area [mm]
            act_thickness_sco_w = sco_vol_w/act_width_mean_sco_w*(d50*1e03) # Scour active thickness V_sco/act_area [mm]
            
            # Append all values in arrays
            tot_vol_w_array = np.append(tot_vol_w_array, tot_vol_w)
            sum_vol_w_array = np.append(sum_vol_w_array, sum_vol_w)
            dep_vol_w_array = np.append(dep_vol_w_array, dep_vol_w)
            sco_vol_w_array = np.append(sco_vol_w_array, sco_vol_w)
            
            morph_act_area_w_array = np.append(morph_act_area_w_array, morph_act_area_w)
            morph_act_area_dep_w_array = np.append(morph_act_area_dep_w_array, morph_act_area_dep_w)
            morph_act_area_sco_w_array = np.append(morph_act_area_sco_w_array, morph_act_area_sco_w)
                           
            act_width_mean_w_array = np.append(act_width_mean_w_array, act_width_mean_w)
            act_width_mean_dep_w_array = np.append(act_width_mean_dep_w_array, act_width_mean_dep_w)
            act_width_mean_sco_w_array = np.append(act_width_mean_sco_w_array, act_width_mean_sco_w)
            
            act_thickness_w_array = np.append(act_thickness_w_array, act_thickness_w)
            act_thickness_dep_w_array = np.append(act_thickness_dep_w_array, act_thickness_dep_w)
            act_thickness_sco_w_array = np.append(act_thickness_sco_w_array, act_thickness_sco_w)
            

        if plot_mode ==2:
            
            # Plots
            fig1, axs = plt.subplots(1,1,dpi=200, sharex=True, tight_layout=True)
            axs.plot(x_data, dep_vol_w_array, '-', c='brown')
            axs.set_title(run)
            axs.set_xlabel('Window analysis length [m]')
            axs.set_ylabel('Deposition volumes V/(W*L*d50) [-]')
            # plt.savefig(os.path.join(plot_dir, run +'_morphW_interp.png'), dpi=200)
            plt.show()
        
            fig1, axs = plt.subplots(1,1,dpi=200, sharex=True, tight_layout=True)
            axs.plot(x_data, sco_vol_w_array, '-', c='brown')
            axs.set_title(run)
            axs.set_xlabel('Window analysis length [m]')
            axs.set_ylabel('Scour volumes V/(W*L*d50) [-]')
            # plt.savefig(os.path.join(plot_dir, run +'_morphW_interp.png'), dpi=200)
            plt.show()
        
            fig1, axs = plt.subplots(1,1,dpi=200, sharex=True, tight_layout=True)
            axs.plot(x_data, act_width_mean_w_array, '-', c='brown')
            axs.set_title(run)
            axs.set_xlabel('Window analysis length [m]')
            axs.set_ylabel('Active width actW/W [-]')
            # plt.savefig(os.path.join(plot_dir, run +'_morphW_interp.png'), dpi=200)
            plt.show()
        
            fig1, axs = plt.subplots(1,1,dpi=200, sharex=True, tight_layout=True)
            axs.plot(x_data, act_thickness_w_array, '-', c='brown')
            axs.set_title(run)
            axs.set_xlabel('Longitudinal coordinate [m]')
            axs.set_ylabel('Active thickness [mm]')
            # plt.savefig(os.path.join(plot_dir, run +'_morphW_interp.png'), dpi=200)
            plt.show()
            
            
    
    # Fixed window without overlapping
    if windows_mode == 2:
        
        # Calculate the number of suitable windows in the channel length
        c_array = []
        W_cols = int(W/px_x*1e03)
        for i in range(1, round(dim_x/W_cols)):
            c = math.floor(dim_x/(W_cols*i))
            if c*W_cols*i<=dim_x:
                c_array = np.append(c_array, c)
            else:
                pass
    
        # Define the components of the slicing operation (exclude the first one)
        f_cols_array = [0,0]
        x_data = [] # X data for the plot
        n = 0 # Initialize variable count
        for m in range(0,len(c_array)):
            # m is the window dimension in columns
            n+=1
            for i in range(1,(math.floor(dim_x/(W_cols*(m+1)))+1)):
                f_cols = [round(W_cols*(m+1)*(i-1), 1), round(W_cols*(m+1)*(i),1)]
                f_cols_array = np.vstack((f_cols_array, f_cols))
                x_data = np.append(x_data, n)
                
        x_data = (x_data)*W
        
        # Resize f_cols_array
        f_cols_array = f_cols_array[1:]
    
        
        for p in range(0, f_cols_array.shape[0]): # Loop over all the available window
            
            w_len = (f_cols_array[p,1] - f_cols_array[p,0])*px_x/1e03 # Define the window lwgth
                                
            # Define total volume matrix, Deposition matrix and Scour matrix
            DoD_vol_w = DoD_vol[:, f_cols_array[p,0]:f_cols_array[p,1]] # Total volume matrix
            dep_DoD_w = dep_DoD[:, f_cols_array[p,0]:f_cols_array[p,1]] # DoD of only deposition data
            sco_DoD_w = sco_DoD[:, f_cols_array[p,0]:f_cols_array[p,1]] # DoD of only scour data
            
            # Define active pixel matrix
            act_px_matrix_w = act_px_matrix[:, f_cols_array[p,0]:f_cols_array[p,1]] # Active pixel matrix, both scour and deposition
            act_px_matrix_dep_w = act_px_matrix_dep[:, f_cols_array[p,0]:f_cols_array[p,1]] # Active deposition matrix 
            act_px_matrix_sco_w = act_px_matrix_sco[:, f_cols_array[p,0]:f_cols_array[p,1]] # Active scour matrix
            
            # Calculate principal quantities:
            # Volumes
            tot_vol_w = np.sum(DoD_vol_w)*px_x*px_y/(W*w_len*d50*1e09)# Total volume as V/(L*W*d50) [-] considering negative sign for scour
            sum_vol_w = np.sum(np.abs(DoD_vol_w))*px_x*px_y/(W*w_len*d50*1e09) # Sum of scour and deposition volume as V/(L*W*d50) [-]
            dep_vol_w = np.sum(dep_DoD_w)*px_x*px_y/(W*w_len*d50*1e09) # Deposition volume as V/(L*W*d50) [-]
            sco_vol_w = np.sum(sco_DoD_w)*px_x*px_y/(W*w_len*d50*1e09) # Scour volume as V/(L*W*d50) [-]
            
            # Areas:
            morph_act_area_w = np.count_nonzero(act_px_matrix_w)*px_x*px_y/(W*w_len*1e06) # Active area both in terms of scour and deposition as A/(W*L) [-]
            morph_act_area_dep_w = np.count_nonzero(act_px_matrix_dep_w)*px_x*px_y/(W*w_len*1e06) # Active deposition area as A/(W*L) [-]
            morph_act_area_sco_w = np.count_nonzero(act_px_matrix_sco_w)*px_x*px_y/(W*w_len*1e06) # Active scour area as A/(W*L) [-]
            
            # Widths:
            act_width_mean_w = np.count_nonzero(act_px_matrix_w)*px_x*px_y/(W*w_len*1e06) # Total mean active width [%] - Wact/W
            act_width_mean_dep_w = np.count_nonzero(act_px_matrix_dep_w)*px_x*px_y/(W*w_len*1e06) # Deposition mean active width [%] - Wact/W
            act_width_mean_sco_w = np.count_nonzero(act_px_matrix_sco_w)*px_x*px_y/(W*w_len*1e06) # Scour mean active width [%] - Wact/W
            
            # Thicknesses:
            act_thickness_w = sum_vol_w/morph_act_area_w*(d50*1e03) # Total active thickness (abs(V_sco) + V_dep)/act_area [mm]
            act_thickness_dep_w = dep_vol_w/morph_act_area_dep_w*(d50*1e03) # Deposition active thickness V_dep/act_area [mm]
            act_thickness_sco_w = sco_vol_w/act_width_mean_sco_w*(d50*1e03) # Scour active thickness V_sco/act_area [mm]
            
            # Append all values in arrays
            tot_vol_w_array = np.append(tot_vol_w_array, tot_vol_w)
            sum_vol_w_array = np.append(sum_vol_w_array, sum_vol_w)
            dep_vol_w_array = np.append(dep_vol_w_array, dep_vol_w)
            sco_vol_w_array = np.append(sco_vol_w_array, sco_vol_w)
            
            morph_act_area_w_array = np.append(morph_act_area_w_array, morph_act_area_w)
            morph_act_area_dep_w_array = np.append(morph_act_area_dep_w_array, morph_act_area_dep_w)
            morph_act_area_sco_w_array = np.append(morph_act_area_sco_w_array, morph_act_area_sco_w)
                           
            act_width_mean_w_array = np.append(act_width_mean_w_array, act_width_mean_w)
            act_width_mean_dep_w_array = np.append(act_width_mean_dep_w_array, act_width_mean_dep_w)
            act_width_mean_sco_w_array = np.append(act_width_mean_sco_w_array, act_width_mean_sco_w)
            
            act_thickness_w_array = np.append(act_thickness_w_array, act_thickness_w)
            act_thickness_dep_w_array = np.append(act_thickness_dep_w_array, act_thickness_dep_w)
            act_thickness_sco_w_array = np.append(act_thickness_sco_w_array, act_thickness_sco_w)
        
            
        if plot_mode ==2:
            # Plots
            fig1, axs = plt.subplots(1,1,dpi=200, sharex=True, tight_layout=True)
            axs.plot(x_data, dep_vol_w_array, 'o', c='brown')
            axs.set_title(run)
            axs.set_xlabel('Window analysis length [m]')
            axs.set_ylabel('Deposition volumes V/(W*L*d50) [-]')
            # plt.savefig(os.path.join(plot_dir, run +'_morphW_interp.png'), dpi=200)
            plt.show()
        
            fig1, axs = plt.subplots(1,1,dpi=200, sharex=True, tight_layout=True)
            axs.plot(x_data, sco_vol_w_array, 'o', c='brown')
            axs.set_title(run)
            axs.set_xlabel('Window analysis length [m]')
            axs.set_ylabel('Scour volumes V/(W*L*d50) [-]')
            # plt.savefig(os.path.join(plot_dir, run +'_morphW_interp.png'), dpi=200)
            plt.show()
        
            fig1, axs = plt.subplots(1,1,dpi=200, sharex=True, tight_layout=True)
            axs.plot(x_data, act_width_mean_w_array, 'o', c='brown')
            axs.set_title(run)
            axs.set_xlabel('Window analysis length [m]')
            axs.set_ylabel('Active width actW/W [-]')
            # plt.savefig(os.path.join(plot_dir, run +'_morphW_interp.png'), dpi=200)
            plt.show()
        
            fig1, axs = plt.subplots(1,1,dpi=200, sharex=True, tight_layout=True)
            axs.plot(x_data, act_thickness_w_array, 'o', c='brown')
            axs.set_title(run)
            axs.set_xlabel('Window analysis length [m]')
            axs.set_ylabel('Active thickness [mm]')
            # plt.savefig(os.path.join(plot_dir, run +'_morphW_interp.png'), dpi=200)
            plt.show()
    
    # Fixed window with overlapping
    if windows_mode == 3:
        
        # Calculate the number of suitable windows in the channel length
        c_array = []
        W_cols = int(W/px_x*1e03) # Minimum windows length WxW dimension in columns
        for i in range(1, math.floor(dim_x/W_cols)+1): # per each windows analysis WxWi
            c = dim_x - W_cols*i
            c_array = np.append(c_array, c) # Contains the number of windows for each dimension WxW*i
        else:
            pass
    
        f_cols_array = [0,0]
        x_data = []
        n = 0
        for m in range(1,int(dim_x/W_cols)+1):
            w_length = m*W_cols # Analysis windows length
            # print(w_length)
            n+=1
            for i in range(0,dim_x): # i is the lower limit of the analysis window
                low_lim = i # Analisys window lower limit
                upp_lim = i + w_length # Analisys window upper limit
                
                if upp_lim<=dim_x:
                    # print(low_lim, upp_lim)
                    # print(i+w_length)
                    f_cols = [low_lim, upp_lim] # Lower and upper boundary of the analysis window
                    f_cols_array = np.vstack((f_cols_array, f_cols))
                    x_data = np.append(x_data, n)
                else:
                    pass
        
        x_data = x_data*W
        # Resize f_cols_array
        f_cols_array = f_cols_array[1:]
        
        
        for p in range(0, f_cols_array.shape[0]):
            
            w_len = (f_cols_array[p,1] - f_cols_array[p,0])*px_x/1e03 # Define the window length
            
            # print()
            # print(f_cols_array[p,:])
            # print(w_len)
            
            # Define total volume matrix, Deposition matrix and Scour matrix
            DoD_vol_w = DoD_vol[:, f_cols_array[p,0]:f_cols_array[p,1]] # Total volume matrix
            dep_DoD_w = dep_DoD[:, f_cols_array[p,0]:f_cols_array[p,1]] # DoD of only deposition data
            sco_DoD_w = sco_DoD[:, f_cols_array[p,0]:f_cols_array[p,1]] # DoD of only scour data
            
            # Define active pixel matrix
            act_px_matrix_w = act_px_matrix[:, f_cols_array[p,0]:f_cols_array[p,1]] # Active pixel matrix, both scour and deposition
            act_px_matrix_dep_w = act_px_matrix_dep[:, f_cols_array[p,0]:f_cols_array[p,1]] # Active deposition matrix 
            act_px_matrix_sco_w = act_px_matrix_sco[:, f_cols_array[p,0]:f_cols_array[p,1]] # Active scour matrix
            
            # Calculate principal quantities:
            # Volumes
            tot_vol_w = np.sum(DoD_vol_w)*px_x*px_y/(W*w_len*d50*1e09)# Total volume as V/(L*W*d50) [-] considering negative sign for scour
            sum_vol_w = np.sum(np.abs(DoD_vol_w))*px_x*px_y/(W*w_len*d50*1e09) # Sum of scour and deposition volume as V/(L*W*d50) [-]
            dep_vol_w = np.sum(dep_DoD_w)*px_x*px_y/(W*w_len*d50*1e09) # Deposition volume as V/(L*W*d50) [-]
            sco_vol_w = np.sum(sco_DoD_w)*px_x*px_y/(W*w_len*d50*1e09) # Scour volume as V/(L*W*d50) [-]
            
            # Areas:
            morph_act_area_w = np.count_nonzero(act_px_matrix_w)*px_x*px_y/(W*w_len*1e06) # Active area both in terms of scour and deposition as A/(W*L) [-]
            morph_act_area_dep_w = np.count_nonzero(act_px_matrix_dep_w)*px_x*px_y/(W*w_len*1e06) # Active deposition area as A/(W*L) [-]
            morph_act_area_sco_w = np.count_nonzero(act_px_matrix_sco_w)*px_x*px_y/(W*w_len*1e06) # Active scour area as A/(W*L) [-]
            
            # Widths:
            act_width_mean_w = np.count_nonzero(act_px_matrix_w)*px_x*px_y/(W*w_len*1e06) # Total mean active width [%] - Wact/W
            act_width_mean_dep_w = np.count_nonzero(act_px_matrix_dep_w)*px_x*px_y/(W*w_len*1e06) # Deposition mean active width [%] - Wact/W
            act_width_mean_sco_w = np.count_nonzero(act_px_matrix_sco_w)*px_x*px_y/(W*w_len*1e06) # Scour mean active width [%] - Wact/W
            
            # Thicknesses:
            act_thickness_w = sum_vol_w/morph_act_area_w*(d50*1e03) # Total active thickness (abs(V_sco) + V_dep)/act_area [mm]
            act_thickness_dep_w = dep_vol_w/morph_act_area_dep_w*(d50*1e03) # Deposition active thickness V_dep/act_area [mm]
            act_thickness_sco_w = sco_vol_w/act_width_mean_sco_w*(d50*1e03) # Scour active thickness V_sco/act_area [mm]
            
            # Append all values in arrays
            tot_vol_w_array = np.append(tot_vol_w_array, tot_vol_w)
            sum_vol_w_array = np.append(sum_vol_w_array, sum_vol_w)
            dep_vol_w_array = np.append(dep_vol_w_array, dep_vol_w)
            sco_vol_w_array = np.append(sco_vol_w_array, sco_vol_w)
            
            morph_act_area_w_array = np.append(morph_act_area_w_array, morph_act_area_w)
            morph_act_area_dep_w_array = np.append(morph_act_area_dep_w_array, morph_act_area_dep_w)
            morph_act_area_sco_w_array = np.append(morph_act_area_sco_w_array, morph_act_area_sco_w)
                           
            act_width_mean_w_array = np.append(act_width_mean_w_array, act_width_mean_w)
            act_width_mean_dep_w_array = np.append(act_width_mean_dep_w_array, act_width_mean_dep_w)
            act_width_mean_sco_w_array = np.append(act_width_mean_sco_w_array, act_width_mean_sco_w)
            
            act_thickness_w_array = np.append(act_thickness_w_array, act_thickness_w)
            act_thickness_dep_w_array = np.append(act_thickness_dep_w_array, act_thickness_dep_w)
            act_thickness_sco_w_array = np.append(act_thickness_sco_w_array, act_thickness_sco_w)
        
        if plot_mode ==2:
            # Plots
            fig1, axs = plt.subplots(1,1,dpi=200, sharex=True, tight_layout=True)
            axs.plot(x_data, dep_vol_w_array, 'o', c='brown', markersize=0.1)
            axs.set_title(run)
            axs.set_xlabel('Window analysis length [m]')
            axs.set_ylabel('Deposition volumes V/(W*L*d50) [-]')
            # plt.savefig(os.path.join(plot_dir, run +'_morphW_interp.png'), dpi=200)
            plt.show()
        
            fig1, axs = plt.subplots(1,1,dpi=200, sharex=True, tight_layout=True)
            axs.plot(x_data, sco_vol_w_array, 'o', c='brown', markersize=0.1)
            axs.set_title(run)
            axs.set_xlabel('Window analysis length [m]')
            axs.set_ylabel('Scour volumes V/(W*L*d50) [-]')
            # plt.savefig(os.path.join(plot_dir, run +'_morphW_interp.png'), dpi=200)
            plt.show()
        
            fig1, axs = plt.subplots(1,1,dpi=200, sharex=True, tight_layout=True)
            axs.plot(x_data, act_width_mean_w_array, 'o', c='brown', markersize=0.1)
            axs.set_title(run)
            axs.set_xlabel('Window analysis length [m]')
            axs.set_ylabel('Active width actW/W [-]')
            # plt.savefig(os.path.join(plot_dir, run +'_morphW_interp.png'), dpi=200)
            plt.show()
        
            fig1, axs = plt.subplots(1,1,dpi=200, sharex=True, tight_layout=True)
            axs.plot(x_data, act_thickness_w_array, 'o', c='brown', markersize=0.1)
            axs.set_title(run)
            axs.set_xlabel('Window analysis length [m]')
            axs.set_ylabel('Active thickness [mm]')
            # plt.savefig(os.path.join(plot_dir, run +'_morphW_interp.png'), dpi=200)
            plt.show()

    if f == DoDs_name_array[0]:
        dep_vol_w_array_all = np.transpose(np.array(dep_vol_w_array))
        sco_vol_w_array_all = np.transpose(np.array(sco_vol_w_array))
    else:
        pass
    
    dep_vol_w_array_all = np.vstack((dep_vol_w_array_all,dep_vol_w_array))
    dep_vol_mean = np.mean(dep_vol_w_array_all, axis=0)
    dep_vol_std = np.std(dep_vol_w_array_all, axis=0)
    
    sco_vol_w_array_all = np.vstack((sco_vol_w_array_all,sco_vol_w_array))
    sco_vol_mean = np.mean(sco_vol_w_array_all, axis=0)
    sco_vol_std = np.std(sco_vol_w_array_all, axis=0)
    
    if windows_mode==2:
        
        # Loop to define the windows to clusterize data
        array = [0]
        num=0
        for n in range(0,len(c_array)):
            num += c_array[n]
            array = np.append(array, num) # Clusterize window dimension
            
        dep_vol_mean = []
        sco_vol_mean = []
        dep_vol_std = []
        sco_vol_std = []

        x_data_full = x_data
        x_data = []
        for n in range(0, len(array)-1):
            x_data = np.append(x_data, x_data_full[int(array[n])])
            
        for n in f_cols_array:    
            dep_vol_mean = np.append(dep_vol_mean, np.mean(dep_vol_w_array_all[:,int(array[n]):int(array[n+1])]))
            sco_vol_mean = np.append(sco_vol_mean, np.mean(sco_vol_w_array_all[:,int(array[n]):int(array[n+1])]))
            dep_vol_std = np.append(dep_vol_std, np.std(dep_vol_w_array_all[:,int(array[n]):int(array[n+1])]))
            sco_vol_std = np.append(sco_vol_std, np.std(sco_vol_w_array_all[:,int(array[n]):int(array[n+1])]))
            # To finish
            
    


            
    if windows_mode == 3:
        # Loop to define the windows to clusterize data
        array = [0]
        num=0
        for n in range(0,len(c_array)):
            num += c_array[n]
            array = np.append(array, num) # Clusterize window dimension
            
        dep_vol_mean = []
        sco_vol_mean = []
        dep_vol_std = []
        sco_vol_std = []

        x_data_full = x_data
        x_data = []
        for n in range(0, len(array)-1):
            # low_lim = int(f_cols_array[n,0])
            # upp_lim = int(f_cols_array[n,1])                  
            x_data = np.append(x_data, round(x_data_full[int(array[n])+n],1))
            # dep_vol_mean = np.append(dep_vol_mean, np.mean(dep_vol_w_array_all[:,low_lim:upp_lim]))
            # sco_vol_mean = np.append(sco_vol_mean, np.mean(sco_vol_w_array_all[:,low_lim:upp_lim]))
            # dep_vol_std = np.append(dep_vol_std, np.std(dep_vol_w_array_all[:,low_lim:upp_lim]))
            # sco_vol_std = np.append(sco_vol_std, np.std(sco_vol_w_array_all[:,low_lim:upp_lim]))
            dep_vol_mean = np.append(dep_vol_mean, np.mean(dep_vol_w_array_all[:,int(array[n]):int(array[n+1])]))
            sco_vol_mean = np.append(sco_vol_mean, np.mean(sco_vol_w_array_all[:,int(array[n]):int(array[n+1])]))
            dep_vol_std = np.append(dep_vol_std, np.std(dep_vol_w_array_all[:,int(array[n]):int(array[n+1])]))
            sco_vol_std = np.append(sco_vol_std, np.std(sco_vol_w_array_all[:,int(array[n]):int(array[n+1])]))
            # print(int(array[n]),int(array[n+1]))
            # TODO To finish
        
    
fig3, axs = plt.subplots(2,1,dpi=80, figsize=(10,6), sharex=True, tight_layout=True)
fig3.suptitle(run + ' - Volume')
axs[0].errorbar(x_data, sco_vol_mean, sco_vol_std, linestyle='--', marker='^', color='red')
# axs[0].set_ylim(bottom=0)
axs[0].set_title('Scour')
# axs[0].set_xlabel()
axs[0].set_ylabel('Scour volume V/(L*W*d50) [-]')
axs[1].errorbar(x_data, dep_vol_mean, dep_vol_std, linestyle='--', marker='^', color='blue')
axs[1].set_ylim(bottom=0)
axs[1].set_title('Deposition')
axs[1].set_xlabel('Analysis window length [m]')
axs[1].set_ylabel('Deposition volume V/(L*W*d50) [-]')
# plt.savefig(os.path.join(plot_dir, run +'dep_scour.png'), dpi=200)
plt.show()
    
# # Plots
# fig1, axs = plt.subplots(1,1,dpi=200, sharex=True, tight_layout=True)
# axs.plot(x_data, dep_vol_w_array, 'o', c='brown')
# axs.set_title(run)
# axs.set_xlabel('Longitudinal coordinate [m]')
# axs.set_ylabel('Deposition volumes V/(W*L*d50) [-]')
# # plt.savefig(os.path.join(plot_dir, run +'_morphW_interp.png'), dpi=200)
# plt.show()

# fig1, axs = plt.subplots(1,1,dpi=200, sharex=True, tight_layout=True)
# axs.plot(x_data, sco_vol_w_array, 'o', c='brown')
# axs.set_title(run)
# axs.set_xlabel('Longitudinal coordinate [m]')
# axs.set_ylabel('Scour volumes V/(W*L*d50) [-]')
# # plt.savefig(os.path.join(plot_dir, run +'_morphW_interp.png'), dpi=200)
# plt.show()

# fig1, axs = plt.subplots(1,1,dpi=200, sharex=True, tight_layout=True)
# axs.plot(x_data, act_width_mean_w_array, 'o', c='brown')
# axs.set_title(run)
# axs.set_xlabel('Longitudinal coordinate [m]')
# axs.set_ylabel('Active width actW/W [-]')
# # plt.savefig(os.path.join(plot_dir, run +'_morphW_interp.png'), dpi=200)
# plt.show()

# fig1, axs = plt.subplots(1,1,dpi=200, sharex=True, tight_layout=True)
# axs.plot(x_data, act_thickness_w_array, 'o', c='brown')
# axs.set_title(run)
# axs.set_xlabel('Longitudinal coordinate [m]')
# axs.set_ylabel('Active thickness [mm]')
# # plt.savefig(os.path.join(plot_dir, run +'_morphW_interp.png'), dpi=200)
# plt.show()