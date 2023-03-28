#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 09:44:30 2021

@author: erri
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

########################################################################################################################
# SETUP FOLDERS
########################################################################################################################
# setup working directory and DEM's name
w_dir = '/home/erri/Documents/morphological_approach/3_output_data/q2.0_2/2_prc_laser/surveys/'
DEM1_name = 'matrix_bed_norm_q20S5.txt'
DEM2_name = 'matrix_bed_norm_q20S6.txt'

# array mask for filtering data outside the channel domain
array_mask, array_mask_path = 'array_mask.txt', '/home/erri/Documents/morphological_approach/2_raw_data'
# TODO Modificare la maschera sulla base dei nuovi output Laser [soglia a 12mm]
path_DEM1 = os.path.join(w_dir, DEM1_name)
path_DEM2 = os.path.join(w_dir, DEM2_name)
DoD_name = 'DoD_' + DEM2_name[19:21] + '-' + DEM1_name[19:21] + '_'

# Output folder
name_out = 'script_outputs_' + DEM2_name[19:21] + '-' + DEM1_name[19:21]
dir_out = '/home/erri/Documents/morphological_approach/3_output_data/q1.0_2/2_prc_laser/DoDs/'
path_out = os.path.join(dir_out, name_out)
if os.path.exists(path_out):
    pass
else:
    os.mkdir(path_out)

########################################################################################################################
# SETUP SCRIPT PARAMETERS
########################################################################################################################
# Thresholds values
thrs_1 = 2.0  # [mm]
thrs_2 = 15.0  # [mm]
neigh_thrs = 4  # [-]

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

    DEM1=DEM1[0:arr_shape[0], 0:arr_shape[1]]
    DEM2=DEM2[0:arr_shape[0], 0:arr_shape[1]]


##############################################################################
# PERFORM DEM OF DIFFERENCE - DEM2-DEM1
##############################################################################

# mask for filtering data outside the channel domain
array_mask = np.loadtxt(os.path.join(array_mask_path, array_mask))
if not(array_mask.shape == arr_shape):
    array_mask=array_mask[0:arr_shape[0], 0:arr_shape[1]]
array_msk = np.where(np.isnan(array_mask), 0, 1)
array_msk_nan = np.where(np.logical_not(np.isnan(array_mask)), 1, np.nan)

# Raster dimension
dim_x, dim_y = DEM1.shape

# Creating DoD array with np.nan
DoD_raw = np.zeros(DEM1.shape)
DoD_raw = np.where(np.logical_or(DEM1 == NaN, DEM2 == NaN), np.nan, DEM2 - DEM1)
# Creating GIS readable DoD array (np.nan as -999)
DoD_raw_rst = np.zeros(DEM1.shape)
DoD_raw_rst = np.where(np.logical_or(DEM1 == NaN, DEM2 == NaN), NaN, DEM2 - DEM1)

# Count the number of pixels in the channel area
DoD_count = np.count_nonzero(np.where(np.isnan(DoD_raw), 0, 1))
print('Active pixels:', DoD_count)

# DoD statistics
print('The minimum DoD value is:\n', np.nanmin(DoD_raw))
print('The maximum DoD value is:\n', np.nanmax(DoD_raw))
print('The DoD shape is:\n', DoD_raw.shape)

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
            w_norm = w / (sum(sum(w)))  # Normalizing ker
            DoD_mean[i, j] = np.nansum(k * w_norm)

# Filtered array weighted average by nan.array mask
DoD_mean_msk = DoD_mean * array_msk_nan
# Create a GIS readable DoD mean (np.nann as -999)
DoD_mean_rst = np.where(np.isnan(DoD_mean_msk), NaN, DoD_mean_msk)


# Filtering data less than thrs_1
mask_thrs_1 = abs(DoD_mean_msk) > thrs_1
DoD_mean_th1 = DoD_mean_msk * mask_thrs_1 # * array_msk_nan
DoD_mean_th1_rst = np.where(np.isnan(DoD_mean_th1), NaN, DoD_mean_th1)



# Neighbourhood coalition analysis
domain_neigh = np.pad(DoD_mean_th1, 1, mode='edge')  # Analysis domain
coal_neigh = np.zeros(DEM1.shape)  # Initialized output array
# TODO Controllare che nessun valore venga escluso da questa analisi
for i in range(0, dim_x):
    for j in range(0, dim_y):
        if np.isnan(DoD_mean_th1[i, j]):
            coal_neigh[i, j] = np.nan
        elif thrs_1 <= abs(DoD_mean_th1[i, j]) <= thrs_2:
            ker = np.array([[domain_neigh[i, j], domain_neigh[i, j + 1], domain_neigh[i, j + 2]],
                            [domain_neigh[i + 1, j], domain_neigh[i + 1, j + 1], domain_neigh[i + 1, j + 2]],
                            [domain_neigh[i + 2, j], domain_neigh[i + 2, j + 1], domain_neigh[i + 2, j + 2]]])
            if DoD_mean_th1[i, j] < 0 and np.count_nonzero(ker < 0) > neigh_thrs:
                coal_neigh[i, j] = DoD_mean_th1[i, j]
            elif DoD_mean_th1[i, j] > 0 and np.count_nonzero(ker > 0) > neigh_thrs:
                coal_neigh[i, j] = DoD_mean_th1[i, j]
            else:
                coal_neigh[i, j] = 0
        else:
            coal_neigh[i,j] = DoD_mean_th1[i,j]
            ...

# Avoiding zero-surrounded pixel
domain_neigh2 = np.pad(coal_neigh, 1, mode='edge')  # Analysis domain
for i in range(0, dim_x):
    for j in range(0,dim_y):
        ker = np.array([[domain_neigh2[i, j], domain_neigh2[i, j + 1], domain_neigh2[i, j + 2]],
                        [domain_neigh2[i + 1, j], 0, domain_neigh2[i + 1, j + 2]],
                        [domain_neigh2[i + 2, j], domain_neigh2[i + 2, j + 1], domain_neigh2[i + 2, j + 2]]])
        num = np.count_nonzero(ker == 0) + np.count_nonzero(~np.isnan(ker))
        if num == 8:
            coal_neigh[i,j]=0
            ...




DoD_out = coal_neigh # * array_msk_nan
# Create a GIS readable filtered DoD (np.nann as -999)
DoD_out_rst = np.where(np.isnan(DoD_out), NaN, DoD_out)

##############################################################################
# PLOT RAW DOD, MEAN DOD AND FILTERED DOD
##############################################################################
# Plot data using nicer colors
colors = ['linen', 'lightgreen', 'darkgreen', 'maroon']
class_bins = [-10.5, -1.5, 0, 1.5, 10.5]
cmap = ListedColormap(colors)
norm = BoundaryNorm(class_bins,
                    len(colors))

fig, (ax1, ax2, ax3) = plt.subplots(3,1)

raw= ax1.imshow(DoD_raw, cmap=cmap, norm=norm)
ax1.set_title('raw DoD')

mean = ax2.imshow(DoD_mean_th1, cmap=cmap, norm=norm)
ax2.set_title('mean DoD')

filt = ax3.imshow(DoD_out, cmap=cmap, norm=norm)
ax3.set_title('Filtered DoD')

#fig.colorbar()
fig.tight_layout()
plt.show()
plt.savefig(path_out + '/raster.pdf') # raster (png, jpg, rgb, tif), vector (pdf, eps), latex (pgf)
#plt.imshow(DoD_out, cmap='RdYlGn')

##############################################################################
# VOLUMES
##############################################################################
# DoD filtered name: coal_neigh
# Create new raster where apply volume calculation
# DoD>0 --> Deposition, DoD<0 --> Scour
DoD_vol = np.where(np.isnan(coal_neigh), 0, coal_neigh)
DEP = (DoD_vol>0)*DoD_vol
SCO = (DoD_vol<0)*DoD_vol
print('Total volume [mm^3]:', np.sum(DoD_vol)*px_x*px_y)
print('Deposition volume [mm^3]:', np.sum(DEP)*px_x*px_y)
print('Scour volume [mm^3]:', np.sum(SCO)*px_x*px_y)


#volume_filt1 = np.sum(np.abs(filtered1_raster_volume))*px_x*px_y
#print('DoD filt_1 volume:', volume_filt1, 'mm^3')

##############################################################################
# SAVE DATA
##############################################################################

#percorso = '/home/erri/Documents/morphological_approach/3_output_data/q1.5/2_prc_laser/script_outputs_s7-s6/verifica/'
#np.savetxt(percorso + 'DoD_raw.txt', DoD_raw, fmt='%0.1f', delimiter='\t')
#np.savetxt(percorso + 'DoD_mean.txt', DoD_mean, fmt='%0.1f', delimiter='\t')

# RAW DoD
# Print raw DoD in txt file (NaN as np.nan)
np.savetxt(path_out + '/' + DoD_name + 'raw.txt', DoD_raw, fmt='%0.1f', delimiter='\t')
# Printing raw DoD in txt file (NaN as -999)
np.savetxt(path_out + '/' + DoD_name + 'raw_rst.txt', DoD_raw_rst, fmt='%0.1f', delimiter='\t')

# MEAN DoD
# Print DoD mean in txt file (NaN as np.nan)
np.savetxt(path_out + '/' + DoD_name + 'DoD_mean.txt', DoD_mean_rst , fmt='%0.1f', delimiter='\t')

# MEAN + THRS1 DoD
# Print DoD mean, threshold 1 filtered in txt file (NaN as np.nan)
np.savetxt(path_out + '/' + DoD_name + 'DoD_mean_th1.txt', DoD_mean_th1, fmt='%0.1f', delimiter='\t')
# # Print filtered DoD (with NaN as -999)
np.savetxt(path_out + '/' + DoD_name + 'DoD_mean_th1_rst.txt', DoD_mean_th1_rst, fmt='%0.1f', delimiter='\t')

#MEAN + THRS_1 + NEIGH ANALYSIS DoD
# Print filtered DoD (with np.nan)...
np.savetxt(path_out + '/' + DoD_name + 'filt_.txt', DoD_out, fmt='%0.1f', delimiter='\t')
# Print filtered DoD (with NaN as -999)
np.savetxt(path_out + 'DoD_name' + 'filt_raw_rst.txt', DoD_out_rst, fmt='%0.1f', delimiter='\t')

# # Print DoD and filtered DoD (with NaN as -999) in a GIS readable format (ASCII grid):
# with open(path_out + '/' + DoD_name + 'header.txt') as f_head:
#     w_header = f_head.read()    # Header
# with open(path_out + '/' + DoD_name + 'raw_rst.txt') as DoD:
#     w_DoD_raw= DoD.read()   # Raw DoD
# with open(path_out + 'DoD_name' + 'filt_raw_rst.txt') as DoD_filt:
#     w_DoD_filt = DoD_filt.read()    # Filtered DoD
# with open(path_out + '/' + DoD_name + 'DoD_mean_th1_rst.txt') as DoD_mn_th1:
#     w_DoD_mean_th1 = DoD_mn_th1.read()
# with open(path_out + '/' + DoD_name + 'DoD_mean.txt') as DoD_mn:
#     w_DoD_mean = DoD_mn.read()    # DoD mean
#     # Print GIS readable raster [raw DoD, mean DOD, filtered DoD]
#     DoD = w_header + w_DoD_raw
#     DoD_mean = w_header + w_DoD_mean
#     DoD_mean_th1 = w_header + w_DoD_mean_th1
#     DoD_filt = w_header + w_DoD_filt

# with open(path_out + '/' +'gis-'+ DoD_name + 'raw.txt', 'w') as fp:
#     fp.write(DoD)
# with open(path_out + '/' + 'gis-' + DoD_name + 'mean.txt', 'w') as fp:
#     fp.write(DoD_mean)
# with open(path_out + '/' + 'gis-' + DoD_name + 'mean_th1.txt', 'w') as fp:
#     fp.write(DoD_mean_th1)
# with open(path_out + '/' + 'gis-' + DoD_name + 'filt.txt', 'w') as fp:
#     fp.write(DoD_filt)




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