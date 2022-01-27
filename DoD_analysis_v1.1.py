#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 09:44:30 2021

@author: erri
"""
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize as opt
# from matplotlib.colors import ListedColormap, BoundaryNorm

start = time.time() # Set initial time
######################################################################################
# FUNCTIONS
######################################################################################
def interpolate(func, xData, yData, ic=None, bounds=(-np.inf, np.inf)):
    # Interpolate data by fitting a given function, then returns the interpolated curve as a 1d array.
    par, covar = opt.curve_fit(func, xData, yData, p0=ic, maxfev=8000, bounds=bounds)
    if len(par) == 2:
        intCurve = func(xData, par[0], par[1])
    elif len(par) == 3:
        intCurve = func(xData, par[0], par[1], par[2])
    elif len(par) == 4:
        intCurve = func(xData, par[0], par[1], par[2], par[3])
    else:
        print("Interpolation failed. The interpolation function must have 2 or 3 parameters")
        intCurve = -1 * np.ones(len(xData))
    return par, intCurve, covar

# Scour and deposition volumes interpolation function
def func(x,A,B):
    y = A*(1-np.exp(-x/B))
    return y

######################################################################################
# SETUP FOLDERS
######################################################################################
# setup working directory and DEM's name
run = 'q07'
home_dir = os.getcwd()
input_dir = os.path.join(home_dir, 'surveys', run)
report_dir = os.path.join(home_dir, 'output')

##############################################################################
# SETUP SCRIPT PARAMETERS
##############################################################################
# MODEs
'''
Mask mode:
    1 = mask the flume edge
    2 = mask the upstream half flume
    3 = mask the downstream half flume
Process mode: (NB: set DEMs name)
    1 = batch process
    2 = single run process
'''
mask_mode=1
process_mode = 1
DEM1_single_name = 'matrix_bed_norm_q07S5.txt' # DEM1 name
DEM2_single_name = 'matrix_bed_norm_q07S6.txt' # DEM2 name

# Thresholds values
thrs_1 = 2.0  # [mm] # Lower threshold
thrs_2 = 15.0  # [mm] # Upper threshold
neigh_thrs = 5  # [-] # Number of neighborhood cells for validation

# Pixel dimension
px_x = 50 # [mm]
px_y = 5 # [mm]

# Not a number raster value (NaN)
NaN = -999

# Run parameters
dt = 47 # dt between runs in minutes (real time)
dt_xnr = 0.5 # temporal discretization in terms of Exner time (Texner between runs)

files=[] # initializing filenames list
# Creating array with file names:
for f in sorted(os.listdir(input_dir)):
    path = os.path.join(input_dir, f)
    if os.path.isfile(path) and f.endswith('.txt') and f.startswith('matrix_bed_norm'):
        files = np.append(files, f)
        
# Initialize arrays
comb = np.array([]) # combination of differences
DoD_count_array=[] # Active pixel
volumes_array=[] # Tot volume
dep_array=[] # Deposition volume
sco_array=[] # Scour volume
active_area_array=[] # Active area 
report_matrix = [] #Report matrix
# matrix_volumes=np.zeros((len(files)-1, len(files)+1)) # Volumes report matrix
matrix_volumes=np.zeros((len(files)-1, len(files)+1)) # Volumes report matrix
# matrix_dep=np.zeros((len(files)-1, len(files)+1)) # Deposition volume report matrix
matrix_dep=np.zeros((len(files)+3, len(files)+1)) # Deposition volume report matrix
# matrix_sco=np.zeros((len(files)-1, len(files)+1)) # Scour volume report matrix
matrix_sco=np.zeros((len(files)+3, len(files)+1)) # Scour volume report matrix

######################################################################################
# SETUP MASKS
######################################################################################
# array mask for filtering data outside the channel domain
#TODO check mask
array_mask_name, array_mask_path = 'array_mask.txt', home_dir
# Load array:
array_mask = np.loadtxt(os.path.join(array_mask_path, array_mask_name))

array_mask = np.where(np.isnan(array_mask), 0, 1) # Convert in mask with 0 and 1
array_mask_nan = np.where(array_mask==0, np.nan, 1) # Convert in mask with np.nan and 1

# Here we can split in two parts the DEMs or keep the entire one
if mask_mode==1:
    pass
elif mask_mode==2: # Working downstream, masking upstream
   array_mask[:,:-int(array_mask.shape[1]/2)] = NaN
   array_mask=np.where(array_mask==NaN, np.nan, array_mask)

elif mask_mode==3: # Working upstream, masking downstream
    array_mask[:,int(array_mask.shape[1]/2):] = NaN
    array_mask=np.where(array_mask==NaN, np.nan, array_mask)
        
######################################################################################
# LOOP OVER ALL DEMs COMBINATIONS
######################################################################################
# Perform difference over all combination of DEMs in the working directory
for h in range (0, len(files)-1):
    for k in range (0, len(files)-1-h):
        DEM1_name=files[h]
        DEM2_name=files[h+1+k]
        comb = np.append(comb, DEM2_name + '-' + DEM1_name)
        
        # write DEM1 and DEM2 names below to avoid batch differences processing 
        if process_mode==1:
            pass
        elif process_mode==2:
            DEM1_name = DEM1_single_name
            DEM2_name = DEM2_single_name
        
        # Specify DEMs path...
        path_DEM1 = os.path.join(input_dir, DEM1_name)
        path_DEM2 = os.path.join(input_dir, DEM2_name)
        # ...and DOD name.
        DoD_name = 'DoD_' + DEM2_name[19:21] + '-' + DEM1_name[19:21] + '_'
        
        # Output folder
        output_name = 'script_outputs_' + DEM2_name[19:21] + '-' + DEM1_name[19:21] # Set outputs name
        output_dir = os.path.join(home_dir, 'DoDs') # Set outputs directory
        path_out = os.path.join(output_dir,  output_name) # Set outputs path
        if not(os.path.exists(output_dir)):
            os.mkdir(output_dir)
        if not(os.path.exists(path_out)):
            os.mkdir(path_out)
                          
                          
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
        
        # Reshaping mask
        if not(array_mask.shape == arr_shape):
            print('mask and array shape not match...')
            print('reshaping mask...')
            print()
            print()
            if array_mask.shape[0]>=arr_shape[0] and array_mask.shape[1]>=arr_shape[1]:
                array_mask=array_mask[0:arr_shape[0], 0:arr_shape[1]] # Reshape mask 0,1
                array_mask_nan=array_mask[0:arr_shape[0], 0:arr_shape[1]] # Reshape mask np.nan,1
            else:
                # TODO optimize for all DoD dimension, non only for q07 runs DoD
                array_mask_reshaped=np.zeros(arr_shape) # Create a zeros array DEM shapes
                array_mask_reshaped[2:-2,:]=array_mask[:, 0:arr_shape[1]]
                array_mask=array_mask_reshaped
                array_mask_nan=np.where(array_mask==0, np.nan, array_mask)
        
        ##############################################################################
        # PERFORM DEM OF DIFFERENCE - DEM2-DEM1
        ##############################################################################
        # Print DoD name
        print(DEM2_name, '-', DEM1_name)
        # Raster dimension
        dim_x, dim_y = DEM1.shape
        
        # Creating DoD array with np.nan
        DoD_raw = np.zeros(DEM1.shape)
        DoD_raw = np.where(np.logical_or(DEM1 == NaN, DEM2 == NaN), np.nan, DEM2 - DEM1)
        # Masking with array mask
        DoD_raw = DoD_raw*array_mask_nan
        # Creating GIS readable DoD array (np.nan as -999)
        DoD_raw_rst = np.zeros(DoD_raw.shape)
        DoD_raw_rst = np.where(np.isnan(DoD_raw), NaN, DoD_raw)
        
        
        # Count the number of pixels in the channel area
        DoD_count = np.count_nonzero(np.where(np.isnan(DoD_raw), 0, 1))
        print('Active pixels:', DoD_count)
        DoD_count_array = np.append(DoD_count_array, DoD_count)
        
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
                    ker1 = np.array([[domain_avg[i, j], domain_avg[i, j + 1], domain_avg[i, j + 2]],
                                  [domain_avg[i + 1, j], domain_avg[i + 1, j + 1], domain_avg[i + 1, j + 2]],
                                  [domain_avg[i + 2, j], domain_avg[i + 2, j + 1], domain_avg[i + 2, j + 2]]])
                    w = np.array([[0, 1, 0],
                                  [0, 2, 0],
                                  [0, 1, 0]])
                    w_norm = w / (sum(sum(w)))  # Normalizing weight matrix
                    DoD_mean[i, j] = np.nansum(ker1 * w_norm)
        #TODO convert Array in a %.1f format
        # # Filtered array weighted average by nan.array mask
        # DoD_mean = DoD_mean * array_msk_nan
        # Create a GIS readable DoD mean (np.nan as -999)
        DoD_mean = np.round(DoD_mean, 1) # Round data to 1 decimal precision
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
                    ker2 = np.array([[DoD_filt_domain[i, j], DoD_filt_domain[i, j + 1], DoD_filt_domain[i, j + 2]],
                                    [DoD_filt_domain[i + 1, j], DoD_filt_domain[i + 1, j + 1], DoD_filt_domain[i + 1, j + 2]],
                                    [DoD_filt_domain[i + 2, j], DoD_filt_domain[i + 2, j + 1], DoD_filt_domain[i + 2, j + 2]]])
                    if not((DoD_filt[i,j] > 0 and np.count_nonzero(ker2 > 0) >= neigh_thrs) or (DoD_filt[i,j] < 0 and np.count_nonzero(ker2 < 0) >= neigh_thrs)):
                        # So if the nature of the selected cell is not confirmed...
                        DoD_filt[i,j] = 0
                        
        DoD_filt = np.round(DoD_filt, 1) # Round data to 1 decimal precision
        # Create a GIS readable filtered DoD (np.nann as -999)
        DoD_filt_rst = np.where(np.isnan(DoD_filt), NaN, DoD_filt)
                        
        # Avoiding zero-surrounded pixel
        DoD_filt_nozero=np.copy(DoD_filt) # Initialize filtered DoD array as a copy of the filtered one
        zerosur_domain = np.pad(DoD_filt_nozero, 1, mode='edge') # Create analysis domain
        for i in range(0,dim_x):
            for j in range(0,dim_y):
                if DoD_filt_nozero[i,j] != 0 and not(np.isnan(DoD_filt_nozero[i,j])): # Limiting the analysis to non-zero numbers 
                    # Create kernel
                    ker3 = np.array([[zerosur_domain[i, j], zerosur_domain[i, j + 1], zerosur_domain[i, j + 2]],
                                    [zerosur_domain[i + 1, j], zerosur_domain[i + 1, j + 1], zerosur_domain[i + 1, j + 2]],
                                    [zerosur_domain[i + 2, j], zerosur_domain[i + 2, j + 1], zerosur_domain[i + 2, j + 2]]])
                    zero_count = np.count_nonzero(ker3 == 0) + np.count_nonzero(np.isnan(ker3))
                    if zero_count == 8:
                        DoD_filt_nozero[i,j] = 0
                    else:
                        pass

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
        # =+SUMIFS(A1:JS144, A1:JS144,">0")*5*50(LibreCalc function)
        DoD_vol = np.where(np.isnan(DoD_filt_nozero), 0, DoD_filt_nozero)
        DEP = (DoD_vol>0)*DoD_vol
        SCO = (DoD_vol<0)*DoD_vol
        
        #Print results:
        print('Total volume [mm^3]:', "{:.1f}".format(np.sum(DoD_vol)*px_x*px_y))
        print('Deposition volume [mm^3]:', "{:.1f}".format(np.sum(DEP)*px_x*px_y))
        print('Scour volume [mm^3]:', "{:.1f}".format(np.sum(SCO)*px_x*px_y))
        
        # Append values to output data array
        volumes_array = np.append(volumes_array, np.sum(DoD_vol)*px_x*px_y)
        dep_array = np.append(dep_array, np.sum(DEP)*px_x*px_y)
        sco_array = np.append(sco_array, np.sum(SCO)*px_x*px_y)
        
        # Active_pixel analysis
        #Resize DoD fpr photos matching
        active_pixel_count = np.where(DoD_vol!=0, 1, 0)
        active_area = np.count_nonzero(active_pixel_count) *px_x*px_y
        print('Area_active: ', "{:.1f}".format(active_area), '[mm**2]')
        active_area_array = np.append(active_area_array, active_area)
        print()
        print()
        
        # Create output matrix as below:
        # DoD step0  1-0   2-1   3-2   4-3   5-4   6-5   7-6   8-7   9-8  average STDEV
        # DoD step1  2-0   3-1   4-2   5-3   6-4   7-5   8-6   9-7        average STDEV
        # DoD step2  3-0   4-1   5-2   6-3   7-4   8-5   9-6              average STDEV
        # DoD step3  4-0   5-1   6-2   7-3   8-4   9-5                    average STDEV
        # DoD step4  5-0   6-1   7-2   8-3   9-4                          average STDEV
        # DoD step5  6-0   7-1   8-2   9-3                                average STDEV
        # DoD step6  7-0   8-1   9-2                                      average STDEV
        # DoD step7  8-0   9-1                                            average STDEV
        # DoD step8  9-0                                                  average STDEV
        #             A     A     A     A     A     A     A     A     A
        #           SD(A) SD(A) SD(A) SD(A) SD(A) SD(A) SD(A) SD(A) SD(A)
        #             B     B     B     B     B     B     B     B     B
        #           SD(B) SD(B) SD(B) SD(B) SD(B) SD(B) SD(B) SD(B) SD(B)
        
        DEM1_num=DEM1_name[20:21]
        DEM2_num=DEM2_name[20:21]
        delta=int(DEM2_name[20:21])-int(DEM1_name[20:21])
        
        if delta != 0:
            # Fill matrix with values
            matrix_volumes[delta-1,h]=np.sum(DoD_vol)*px_x*px_y
            matrix_dep[delta-1,h]=np.sum(DEP)*px_x*px_y
            matrix_sco[delta-1,h]=np.sum(SCO)*px_x*px_y
            # Fill last two columns with AVERAGE and STDEV
            matrix_volumes[delta-1,-2]=np.average(matrix_volumes[delta-1,:len(files)-delta])
            matrix_dep[delta-1,-2]=np.average(matrix_dep[delta-1,:len(files)-delta])
            matrix_sco[delta-1,-2]=np.average(matrix_sco[delta-1,:len(files)-delta])
            matrix_volumes[delta-1,-1]=np.std(matrix_volumes[delta-1,:len(files)-delta])
            matrix_dep[delta-1,-1]=np.std(matrix_dep[delta-1,:len(files)-delta])
            matrix_sco[delta-1,-1]=np.std(matrix_sco[delta-1,:len(files)-delta])
            
        else:
            pass
        
        # Stack consecutive DoDs in a 3D array
        if h==0 and k==0: # initialize the first array with the DEM shape
            DoD_stack = np.zeros([len(files)-1, dim_x, dim_y])
        else:
            pass
        
        if delta==1:
            DoD_stack[h,:,:] = DoD_filt_nozero_rst[:,:]
            
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
        np.savetxt(path_out + '/' + DoD_name + 'nozero.txt', DoD_filt_nozero, fmt='%0.1f', delimiter='\t')
        # Print filtered DoD (with NaN as -999)
        np.savetxt(path_out + '/' + DoD_name + 'filt_nozero_rst.txt', DoD_filt_nozero_rst, fmt='%0.1f', delimiter='\t')
        
        # Print DoD and filtered DoD (with NaN as -999) in a GIS readable format (ASCII grid):
        with open(path_out + '/' + DoD_name + 'header.txt') as f_head:
            w_header = f_head.read()    # Header
        with open(path_out + '/' + DoD_name + 'raw_rst.txt') as f_DoD:
            w_DoD_raw= f_DoD.read()   # Raw DoD
        with open(path_out + '/' + DoD_name + 'mean_rst.txt') as f_DoD_mean:
            w_DoD_mean = f_DoD_mean.read()    # Mean DoD
        with open(path_out + '/' + DoD_name + 'filt_rst.txt') as f_DoD_filt:
            w_DoD_filt = f_DoD_filt.read()    # Filtered DoD
        with open(path_out + '/' + DoD_name + 'filt_nozero_rst.txt') as f_DoD_filt_nozero:
            w_DoD_filt_nozero = f_DoD_filt_nozero.read()    # Avoided zero surrounded pixel DoD
            
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
        
# Print the last DoD outcome
fig, ax = plt.subplots()
im = ax.imshow(np.where(DoD_filt_nozero_rst==NaN, np.nan, DoD_filt_nozero_rst), cmap='RdBu',  vmin=-25, vmax=25)
plt.colorbar(im)
plt.title(DoD_name[:-1], fontweight='bold')
plt.show()

###############################################################################
# VOLUME INTERPOLATION PARAMETER
###############################################################################
'''
Interpolation performed all over the volume data.
Standard deviation is then applied to function parameters
'''

xData=[]
yData_dep=[]
yData_sco=[]
for i in range(0,len(files)-1):
    xData=np.append(xData, np.ones(len(files)-i-1)*(i+1)*dt) # Create xData array for all the volume points
    yData_dep=np.append(yData_dep, matrix_dep[i,:len(files)-i-1]) # deposition volumes (unroll yData)
    yData_sco=np.append(yData_sco, abs(matrix_sco[i,:len(files)-i-1])) # scour volumes (unroll yData)

ic_dep=np.array([np.mean(yData_dep),np.min(xData)]) # Initial deposition parameter guess
ic_sco=np.array([np.mean(yData_sco),np.min(xData)]) # Initial scour parameter guess
par_dep, intCurve_dep, covar_dep = interpolate(func, xData, yData_dep, ic_dep)
par_sco, intCurve_sco, covar_sco = interpolate(func, xData, yData_sco, ic_sco)

# Print scour and deposition matrix the interpolation parameters of the all data interpolation
# matrix_dep[len(files)]

print()
print('All points interpolation parameters:')
print('Deposition interpolation parameters')
print('A=', par_dep[0], 'Variance=', covar_dep[0,0])
print('B=', par_dep[1], 'Variance=', covar_dep[1,1])
print('Scour interpolation parameters')
print('A=', par_sco[0], 'Variance=', covar_sco[0,0])
print('B=', par_sco[1], 'Variance=', covar_sco[1,1])

fig1, axs = plt.subplots(2,1,dpi=100, sharex=True, tight_layout=True)
axs[0].plot(xData, yData_dep, 'o')
axs[0].plot(xData, intCurve_dep, c='red')
axs[0].set_title('Deposition volumes interpolation')
axs[0].set_xlabel('xData')
axs[0].set_ylabel('yData')
axs[1].plot(xData, yData_sco, 'o')
axs[1].plot(xData, intCurve_sco, c='red')
axs[1].set_title('Scour volumes interpolation')
axs[1].set_xlabel('xData')
axs[1].set_ylabel('yData')
plt.show()



for i in range(0, len(files)-3): # Last three columns have 1 or 2 or 3 values: not enought -> interpolation skipped
    xData = np.arange(0, len(files)-i-1, 1)
    
    #Fill deposition matrix
    yData=np.absolute(matrix_dep[:len(files)-i-1,i])
    par, intCurve, covar = interpolate(func, xData, yData)
    matrix_dep[-4,i],  matrix_dep[-2,i]=  par[0], par[1] # Parameter A and B
    matrix_dep[-3,i],  matrix_dep[-1,i]=  covar[0,0], covar[1,1] # STD(A) and STD(B)
    
    # Fill scour matrix
    yData=np.multiply(np.absolute(matrix_sco[:len(files)-i-1,i]), 1/np.max(np.absolute(matrix_sco[:len(files)-i-1,i])))
    par, intCurve, covar = interpolate(func, xData, yData)
    matrix_sco[-4,i],  matrix_sco[-2,i]=  par[0], par[1] # Parameter A and B
    matrix_sco[-3,i],  matrix_sco[-1,i]=  covar[0,0], covar[1,1] # STD(A) and STD(B)

###############################################################################
# SAVE DATA MATRIX
###############################################################################
# Create report matrix
report_matrix = np.array(np.transpose(np.stack((comb, DoD_count_array, volumes_array, dep_array, sco_array, active_area_array))))
report_header = 'DoD_combination, Active pixels, Total volume [mm^3], Deposition volume [mm^3], Scour volume [mm^3], Active area [mm^2]'

report_name = run + '_report.txt'
with open(os.path.join(report_dir , report_name), 'w') as fp:
    fp.write(report_header)
    fp.writelines(['\n'])
    for i in range(0,len(report_matrix[:,0])):
        for j in range(0, len(report_matrix[0,:])):
            if j == 0:
                fp.writelines([report_matrix[i,j]+', '])
            else:
                fp.writelines(["%.3f, " % float(report_matrix[i,j])])
        fp.writelines(['\n'])
fp.close()

# Create deposition report matrix
report_dep_name = os.path.join(report_dir, run +'_dep_report.txt')
np.savetxt(report_dep_name, matrix_dep, fmt='%.3f', delimiter=',', newline='\n')

# Create scour report matrix
report_sco_name = os.path.join(report_dir, run +'_sco_report.txt')
np.savetxt(report_sco_name, matrix_sco, fmt='%.3f', delimiter=',', newline='\n')


# Print scour volumes over increasing timestep:
fig1, ax1 = plt.subplots()
# ax1.bar(np.arange(0, len(matrix_sco[:,0]), 1),abs(matrix_sco[:,0]))
# ax1.plot(t[int(len(t)/10):-int(len(t)/10)], m*t[int(len(t)/10):-int(len(t)/10)]+q)
xData=np.arange(0, len(files)-1, 1)
yData=np.absolute(matrix_sco[:len(files)-1,0])
yError=matrix_sco[:len(files)-1,-1]
ax1.errorbar(xData,yData, yError, linestyle='--', marker='^')
ax1.set_ylim(bottom=0)
ax1.set_title(run)
ax1.set_xlabel('Time')
ax1.set_ylabel('yData')
plt.show()

end = time.time()
print()
print('Execution time: ', (end-start), 's')
