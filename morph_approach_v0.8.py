#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 09:44:30 2021

@author: erri
"""
import os
import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.colors import ListedColormap, BoundaryNorm

######################################################################################
# SETUP FOLDERS
######################################################################################
# setup working directory and DEM's name
# home_dir = '/home/erri/Documents/morphological_approach/3_output_data/q1.0_2_test/2_prc_laser/'
home_dir = '/home/erri/Documents/PhD/Research/2_raw_data/repeat_surveys_test/' # repeat_surveys_test
input_dir = os.path.join(home_dir, 'surveys')
# array mask for filtering data outside the channel domain
array_mask_name, array_mask_path = 'array_mask.txt', '/home/erri/Documents/morphological_approach/2_raw_data'
# TODO Check mask
# TODO Modificare la maschera sulla base dei nuovi output Laser [soglia a 12mm]
files=[] # initializing filenames list
# Creating array with file names:
for f in sorted(os.listdir(input_dir)):
    path = os.path.join(input_dir, f)
    if os.path.isfile(path) and f.endswith('.txt') and f.startswith('matrix_bed_norm'):
        files = np.append(files, f)
# Perform difference over all combination of DEMs in the working directory
comb = [] # combination of differences

for h in range (0, len(files)-1):
    for k in range (0, len(files)-1-h):
        DEM1_name=files[h]
        DEM2_name=files[h+1+k]
        print(DEM2_name, '-', DEM1_name)
        comb = np.append(comb, DEM2_name + '-' + DEM1_name)
        
        # write DEM1 and DEM2 names below to avoid batch differences processing        
        DEM1_name = 'matrix_bed_norm_q07S5same.txt'
        DEM2_name = 'matrix_bed_norm_q07S6same.txt'
        
        
        # Specify DEMs path...
        path_DEM1 = os.path.join(input_dir, DEM1_name)
        path_DEM2 = os.path.join(input_dir, DEM2_name)
        # ...and DOD name.
        DoD_name = 'DoD_' + DEM2_name[19:21] + '-' + DEM1_name[19:21] + '_'
        
        # Output folder
        output_name = 'script_outputs_' + DEM2_name[19:21] + '-' + DEM1_name[19:21] # Set outputs name
        output_dir = os.path.join(home_dir, 'DoDs_0.8') # Set outputs directory
        path_out = os.path.join(output_dir,  output_name) # Set outputs path
        if os.path.exists(path_out): # Check if outputs path already exists
            pass
        else:
            os.mkdir(output_dir)
            os.mkdir(path_out)
        
        ##############################################################################
        # SETUP SCRIPT PARAMETERS
        ##############################################################################
        # Thresholds values
        thrs_1 = 2.0  # [mm] # Lower threshold
        thrs_2 = 15.0  # [mm] # Upper threshold
        neigh_thrs = 5  # [-] # Number of neighborhood cells for validation
        
        # Pixel dimension
        px_x = 50 # [mm]
        px_y = 5 # [mm]
        
        # Not a number raster value (NaN)
        NaN = -999
        
        ##############################################################################
        # DATA READING...
        ##############################################################################
        # Header initialization and extraction
        lines = []
        header = []
        
        with open(path_DEM1, 'r') as file:
            for line in file:
                lines.append(line)  # lines is a list. Each item is a row of the input file
            # Header extraction...
            for i in range(0, 7):
                header.append(lines[i])
        # Header printing in a file txt called header.txt
        with open(path_out + '/' + DoD_name + 'header.txt', 'w') as head:
            head.writelines(header)
        
        ##############################################################################
        # DATA LOADING...
        ##############################################################################
        DEM1 = np.loadtxt(path_DEM1,
                          # delimiter=',',
                          skiprows=8
                          )
        DEM2 = np.loadtxt(path_DEM2,
                          # delimiter=',',
                          skiprows=8)
        
        # Shape control:
        arr_shape=min(DEM1.shape, DEM2.shape)
        if not(DEM1.shape == DEM2.shape):
            print('Attention: DEMs have not the same shape.')
            # reshaping:
            rows = min(DEM1.shape[0], DEM2.shape[0])
            cols = min(DEM1.shape[1], DEM2.shape[1])
            arr_shape = [rows, cols]
        # and reshaping...
            DEM1=DEM1[0:arr_shape[0], 0:arr_shape[1]]
            DEM2=DEM2[0:arr_shape[0], 0:arr_shape[1]]
        
        # Loading mask
        array_mask = np.loadtxt(os.path.join(array_mask_path, array_mask_name))
        # Reshaping mask
        if not(array_mask.shape == arr_shape):
            array_mask=array_mask[0:arr_shape[0], 0:arr_shape[1]]
        array_msk = np.where(np.isnan(array_mask), 0, 1)
        array_msk_nan = np.where(np.logical_not(np.isnan(array_mask)), 1, np.nan)
        
        ##############################################################################
        # PERFORM DEM OF DIFFERENCE - DEM2-DEM1
        ##############################################################################
        
        # Raster dimension
        dim_x, dim_y = DEM1.shape
        
        # Creating DoD array with np.nan
        DoD_raw = np.zeros(DEM1.shape)
        DoD_raw = np.where(np.logical_or(DEM1 == NaN, DEM2 == NaN), np.nan, DEM2 - DEM1)
        # Masking with array mask
        DoD_raw = DoD_raw*array_msk_nan
        # Creating GIS readable DoD array (np.nan as -999)
        DoD_raw_rst = np.zeros(DoD_raw.shape)
        DoD_raw_rst = np.where(np.isnan(DoD_raw), NaN, DoD_raw)
        
        
        # Count the number of pixels in the channel area
        DoD_count = np.count_nonzero(np.where(np.isnan(DoD_raw), 0, 1))
        print('Active pixels:', DoD_count)
        
        # DoD statistics
        # print('The minimum DoD value is:\n', np.nanmin(DoD_raw))
        # print('The maximum DoD value is:\n', np.nanmax(DoD_raw))
        # print('The DoD shape is:\n', DoD_raw.shape)
        
        ##############################################################################
        # DATA FILTERING...
        ##############################################################################
        
        # Perform domain-wide average
        domain_avg = np.pad(DoD_raw, 1, mode='edge') # i size pad with edge values domain
        DoD_mean = np.zeros(DEM1.shape)
        for i in range (0, dim_x):
            for j in range (0, dim_y):
                if np.isnan(DoD_raw[i, j]):
                    DoD_mean[i, j] = np.nan
                else:
                    k = np.array([[domain_avg[i, j], domain_avg[i, j + 1], domain_avg[i, j + 2]],
                                  [domain_avg[i + 1, j], domain_avg[i + 1, j + 1], domain_avg[i + 1, j + 2]],
                                  [domain_avg[i + 2, j], domain_avg[i + 2, j + 1], domain_avg[i + 2, j + 2]]])
                    w = np.array([[0, 1, 0],
                                  [0, 2, 0],
                                  [0, 1, 0]])
                    w_norm = w / (sum(sum(w)))  # Normalizing weight matrix
                    DoD_mean[i, j] = np.nansum(k * w_norm)
        
        # # Filtered array weighted average by nan.array mask
        # DoD_mean = DoD_mean * array_msk_nan
        # Create a GIS readable DoD mean (np.nan as -999)
        DoD_mean_rst = np.where(np.isnan(DoD_mean), NaN, DoD_mean)
        
        
        # Threshold and Neighbourhood analysis process
        DoD_filt = np.copy(DoD_mean) # Initialize filtered DoD array as a copy of the averaged one
        DoD_filt_domain = np.pad(DoD_filt, 1, mode='edge') # Create neighbourhood analysis domain
        
        for i in range(0,dim_x):
            for j in range(0,dim_y):
                if abs(DoD_filt[i,j]) < thrs_1: # Set as "no variation detected" all variations lower than thrs_1 
                    DoD_filt[i,j] = 0
                if abs(DoD_filt[i,j]) >= thrs_1 and abs(DoD_filt[i,j]) <= thrs_2: # Perform neighbourhood analysis for variations between thrs_1 and thrs_2
                # Create kernel
                    ker = np.array([[DoD_filt_domain[i, j], DoD_filt_domain[i, j + 1], DoD_filt_domain[i, j + 2]],
                                    [DoD_filt_domain[i + 1, j], DoD_filt_domain[i + 1, j + 1], DoD_filt_domain[i + 1, j + 2]],
                                    [DoD_filt_domain[i + 2, j], DoD_filt_domain[i + 2, j + 1], DoD_filt_domain[i + 2, j + 2]]])
                    if not((DoD_filt[i,j] > 0 and np.count_nonzero(ker > 0) >= neigh_thrs) or (DoD_filt[i,j] < 0 and np.count_nonzero(ker < 0) >= neigh_thrs)):
                        # So if the nature of the selected cell is not confirmed...
                        DoD_filt[i,j] = 0
                        
        # DoD_out = DoD_filt # * array_msk_nan
        # Create a GIS readable filtered DoD (np.nann as -999)
        DoD_filt_rst = np.where(np.isnan(DoD_filt), NaN, DoD_filt)
                        
        # Avoiding zero-surrounded pixel
        DoD_filt_nozero=np.copy(DoD_filt) # Initialize filtered DoD array as a copy of the filtered one
        zerosur_domain = np.pad(DoD_filt_nozero, 1, mode='edge') # Create analysis domain
        for i in range(0,dim_x):
            for j in range(0,dim_y):
                if DoD_filt_nozero[i,j] != 0 and not(np.isnan(DoD_filt_nozero[i,j])): # Limiting the analysis to non-zero numbers 
                    # Create kernel
                    ker = np.array([[zerosur_domain[i, j], zerosur_domain[i, j + 1], zerosur_domain[i, j + 2]],
                                    [zerosur_domain[i + 1, j], zerosur_domain[i + 1, j + 1], zerosur_domain[i + 1, j + 2]],
                                    [zerosur_domain[i + 2, j], zerosur_domain[i + 2, j + 1], zerosur_domain[i + 2, j + 2]]])
                    zero_count = np.count_nonzero(ker == 0) + np.count_nonzero(np.isnan(ker))
                    if zero_count == 8:
                        DoD_filt_nozero[i,j] = 0
                    else:
                        pass
        
        # Masking DoD_filt_nozero to avoid 
        
        # Create GIS-readable DoD filtered and zero-surrounded avoided
        DoD_filt_nozero_rst = np.where(np.isnan(DoD_filt_nozero), NaN, DoD_filt_nozero)
        
        
        '''
        Output files:
            DoD_raw: it's just the dem of difference, so DEM2-DEM1
            DoD_raw_rst: the same for DoD_raw, but np.nan=Nan
            DoD_mean: DoD_raw with a smoothing along the Y axes, see the weight in the averaging process
            DoD_mean_rst: the same for DoD_mean but np.nan=Nan
            DoD_filt: DoD_mean with a neighbourhood analysis applie
            DoD_filt_rst: the same for DoD_filt but np.nan=Nan
            DoD_filt_nozero: DoD_filt with an avoiding zero-surrounded process applied
            DoD_filt_nozero_rst: the same for DoD_filt_nozero but with np.nan=NaN
        '''
        
        

        
        ##############################################################################
        # PLOT RAW DOD, MEAN DOD AND FILTERED DOD
        ##############################################################################
        # # Plot data using nicer colors
        # colors = ['linen', 'lightgreen', 'darkgreen', 'maroon']
        # class_bins = [-10.5, -1.5, 0, 1.5, 10.5]
        # cmap = ListedColormap(colors)
        # norm = BoundaryNorm(class_bins,
        #                     len(colors))
        
        # fig, (ax1, ax2, ax3) = plt.subplots(3,1)
        
        # raw= ax1.imshow(DoD_raw, cmap=cmap, norm=norm)
        # ax1.set_title('raw DoD')
        
        # mean = ax2.imshow(DoD_mean_th1, cmap=cmap, norm=norm)
        # ax2.set_title('mean DoD')
        
        # filt = ax3.imshow(DoD_out, cmap=cmap, norm=norm)
        # ax3.set_title('Filtered DoD')
        
        # #fig.colorbar()
        # fig.tight_layout()
        # plt.show()
        # plt.savefig(path_out + '/raster.pdf') # raster (png, jpg, rgb, tif), vector (pdf, eps), latex (pgf)
        # #plt.imshow(DoD_out, cmap='RdYlGn')
        
        ##############################################################################
        # VOLUMES
        ##############################################################################
        # DoD filtered name: DoD_filt
        # Create new raster where apply volume calculation
        # DoD>0 --> Deposition, DoD<0 --> Scour
        DoD_vol = np.where(np.isnan(DoD_filt_nozero), 0, DoD_filt_nozero)
        DEP = (DoD_vol>0)*DoD_vol
        SCO = (DoD_vol<0)*DoD_vol
        # print(DEM2_name + '-' + DEM1_name)
        print('Total volume [mm^3]:', np.sum(DoD_vol)*px_x*px_y)
        print('Deposition volume [mm^3]:', np.sum(DEP)*px_x*px_y)
        print('Scour volume [mm^3]:', np.sum(SCO)*px_x*px_y)
        print()
        
        # Active_pixel analysis
        #Resize DoD fpr photos matching
        active_pixel_count = DoD_vol[:,153:]
        active_pixel_count = np.where(active_pixel_count!=0, 1, 0)
        active_area = np.count_nonzero(active_pixel_count) *px_x*px_y
        print('Area_active: ', active_area, '[mm**2]')
        print()
        print()
        
        

        ##############################################################################
        # SAVE DATA
        ##############################################################################
        
        # RAW DoD
        # Print raw DoD in txt file (NaN as np.nan)
        np.savetxt(path_out + '/' + DoD_name + 'raw.txt', DoD_raw, fmt='%0.1f', delimiter='\t')
        # Printing raw DoD in txt file (NaN as -999)
        np.savetxt(path_out + '/' + DoD_name + 'raw_rst.txt', DoD_raw_rst, fmt='%0.1f', delimiter='\t')
        
        # MEAN DoD
        # Print DoD mean in txt file (NaN as np.nan)
        np.savetxt(path_out + '/' + DoD_name + 'mean.txt', DoD_mean , fmt='%0.1f', delimiter='\t')
        # Print filtered DoD (with NaN as -999)
        np.savetxt(path_out + '/' + DoD_name + 'mean_rst.txt', DoD_mean_rst , fmt='%0.1f', delimiter='\t')
                
        # FILTERED DoD
        # Print filtered DoD (with np.nan)...
        np.savetxt(path_out + '/' + DoD_name + 'filt_.txt', DoD_filt, fmt='%0.1f', delimiter='\t')
        # Print filtered DoD (with NaN as -999)
        np.savetxt(path_out + '/' + DoD_name + 'filt_rst.txt', DoD_filt_rst, fmt='%0.1f', delimiter='\t')
        
        # AVOIDED ZERO SURROUNDED DoD
        # Print filtered DoD (with np.nan)...
        np.savetxt(path_out + '/' + DoD_name + 'nozero_.txt', DoD_filt_nozero, fmt='%0.1f', delimiter='\t')
        # Print filtered DoD (with NaN as -999)
        np.savetxt(path_out + '/' + DoD_name + 'filt_nozero_rst.txt', DoD_filt_nozero_rst, fmt='%0.1f', delimiter='\t')
        
        
        
        
        
        # Print DoD and filtered DoD (with NaN as -999) in a GIS readable format (ASCII grid):
        with open(path_out + '/' + DoD_name + 'header.txt') as f_head:
            w_header = f_head.read()    # Header
        with open(path_out + '/' + DoD_name + 'raw_rst.txt') as DoD:
            w_DoD_raw= DoD.read()   # Raw DoD
        with open(path_out + '/' + DoD_name + 'mean_rst.txt') as DoD_mean:
            w_DoD_mean = DoD_mean.read()    # Mean DoD
        with open(path_out + '/' + DoD_name + 'filt_rst.txt') as DoD_filt:
            w_DoD_filt = DoD_filt.read()    # Filtered DoD
        with open(path_out + '/' + DoD_name + 'filt_nozero_rst.txt') as DoD_filt_nozero:
            w_DoD_filt_nozero = DoD_filt_nozero.read()    # Avoided zero surrounded pixel DoD
            
            # Print GIS readable raster [raw DoD, mean DoD, filtered DoD]
            DoD_raw_gis = w_header + w_DoD_raw
            DoD_mean_gis = w_header + w_DoD_mean
            DoD_filt_gis = w_header + w_DoD_filt
            DoD_filt_nozero_gis = w_header + w_DoD_filt_nozero
        
        with open(path_out + '/' +'gis-'+ DoD_name + 'raw.txt', 'w') as fp:
            fp.write(DoD_raw_gis)
        with open(path_out + '/' +'gis-'+ DoD_name + 'mean.txt', 'w') as fp:
            fp.write(DoD_mean_gis)
        with open(path_out + '/' + 'gis-' + DoD_name + 'filt.txt', 'w') as fp:
            fp.write(DoD_filt_gis)
        with open(path_out + '/' + 'gis-' + DoD_name + 'filt_nozero_rst.txt', 'w') as fp:
            fp.write(DoD_filt_nozero_gis)
        
        
        # Cross section analysis
        #n_cross=1
        #y_values = np.arange(0,144*5,5)
        #cross_sct = DoD_out[:,n_cross]
        #fig, ax = plt.subplots(figsize=(20,5))
        #ax.plot(y_values, cross_sct)
        #title = 'Section_'+str(n_cross)
        #ax.set(xlabel='Cross section coordinates [mm]',
        #       ylabel='Elevation [mm]',
        #       title=title)
        #ax.grid()
        #fig.savefig(path_out+'/section'+n_cross+'.png')
        #plt.show()