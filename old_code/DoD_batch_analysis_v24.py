#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 09:44:30 2021

@author: erri
"""
import os
import time
import math
import numpy as np
import matplotlib.pyplot as plt
from DoD_analysis_functions import *
from DoD_analysis_functions_2 import *
from morph_quantities_func_v2 import morph_quantities
from matplotlib.colors import ListedColormap, BoundaryNorm

###############################################################################
# TODO
###############################################################################
# 1. 

start = time.time() # Set initial time

###############################################################################
# SETUP SCRIPT PARAMETERS and RUN MODE
###############################################################################

'''
run mode:
    1 = one run at time
    2 = bath process
data_interpolatuon_mode:
    0 = no interpolation
    1 = data interpolation
windows_mode:
    0 = fixed windows (all the channel)
    1 = floating windows
    2 = fixed windows (WxW, Wx2W, Wx3W, ...) without overlapping
    3 = fixed windows (WxW, Wx2W, Wx3W, ...) with overlapping
mask mode:
    1 = mask the flume edge
    2 = mask the upstream half flume
    3 = mask the downstream half flume
process mode: (NB: set DEMs name)
    1 = batch process
    2 = single run process
save mode:
    0 = save only reports
    1 = save all chart and figure
'''
run_mode = 1
data_interpolation_mode = 0
windows_mode = 3
mask_mode = 1
process_mode = 2
save_plot_mode = 1

# SINGLE RUN NAME
run = 'q10_1'

# Set DEM single name to perform process to specific DEM
DEM1_single_name = 'matrix_bed_norm_' + run +'s'+'0'+'.txt' # DEM1 name
DEM2_single_name = 'matrix_bed_norm_' + run +'s'+'1'+'.txt' # DEM2 name

# Filtering process thresholds values
thrs_zeros = 7 # [-] isolated_killer function threshold
thrs_nature = 5 # [-] nature_checker function threshold
thrs_fill = 7 # [-] isolated_filler function threshold
thrs_1 = 2.0  # [mm] # Lower threshold
thrs_2 = 15.0  # [mm] # Upper threshold

# Survey pixel dimension
px_x = 50 # [mm]
px_y = 5 # [mm]

# Not a number raster value (NaN)
NaN = -999

#%%
###############################################################################
# SETUP FOLDERS and RUNS
###############################################################################
# setup working directory and DEM's name
home_dir = os.getcwd()
DoDs_dir = os.path.join(home_dir, 'DoDs')
report_dir = os.path.join(home_dir, 'output')
plot_dir = os.path.join(home_dir, 'plot')
run_dir = os.path.join(home_dir, 'surveys')

# Save a report with xData as real time in minutes and the value of scour and deposition volumes for each runs
# Check if the file already exists
if os.path.exists(os.path.join(report_dir, 'volume_over_time.txt')):
    os.remove(os.path.join(report_dir, 'volume_over_time.txt'))
else:
    pass

# Create the run name list
RUNS=[]
if run_mode ==2: # batch run mode
    for RUN in sorted(os.listdir(run_dir)): # loop over surveys directories
        if RUN.startswith('q'): # Consider only folder names starting wit q
            RUNS = np.append(RUNS, RUN) # Append run name at RUNS array
elif run_mode==1: # Single run mode
    RUNS=run.split() # RUNS as a single entry array, provided by run variable

#%%
###############################################################################
# INITIALIZE OVERALL REPORT ARRAYS
###############################################################################

# Define volume time scale report matrix:
# B_dep, SD(B_dep), B_sco, SD(B_sco)
volume_temp_scale_report=np.zeros((len(RUNS), 4))

# Define morphW time scale report matrix:
# B_morphW [min], SD(B_morphW)
morphW_temp_scale_report = np.zeros((len(RUNS), 2))

# Define Engelund Gauss model report matrix:
# D [m], Q [m^3/s], Wwet/W [-]
engelund_model_report=np.zeros((len(RUNS),3))

# Array that collect all the morphWact_array dimension.
# It will be used to create the morphWact_matrix
morphWact_dim = [] # Array with the dimensions of morphWact_values array

DoD_length_array=[] # DoD length array

#%%
###############################################################################
# MAIN LOOP OVER RUNS
###############################################################################
for run in RUNS:

    ###########################################################################
    # SETUP FOLDERS
    ###########################################################################
    print('######')
    print(run)
    print('######')
    print()
    # setup working directory and DEM's name
    input_dir = os.path.join(home_dir, 'surveys', run)

    # CREATE FOLDERS
    if not(os.path.exists(report_dir)):
        os.mkdir(report_dir)
    if not(os.path.exists(DoDs_dir)):
        os.mkdir(DoDs_dir)
    if not(os.path.exists(os.path.join(DoDs_dir, 'DoDs_stack'))):
        os.mkdir(os.path.join(DoDs_dir, 'DoDs_stack'))
    if not(os.path.exists(plot_dir)):
        os.mkdir(plot_dir)


    # IMPORT RUN PARAMETERS from file parameters.txt
    # variable run must be as 'q' + discharge + '_' repetition number
    # Parameters.txt structure:
    # discharge [l/s],repetition,run time [min],Texner discretization [-], Channel width [m], slope [m/m]
    # Load parameter matrix:
    parameters = np.loadtxt(os.path.join(home_dir, 'parameters.txt'),
                            delimiter=',',
                            skiprows=1)
    # Extract run parameter depending by run name
    run_param = parameters[np.intersect1d(np.argwhere(parameters[:,1]==float(run[-1:])),np.argwhere(parameters[:,0]==float(run[1:3])/10)),:]

    # Run time data
    dt = run_param[0,2] # dt between runs [min] (real time)
    dt_xnr = run_param[0,3] # temporal discretization in terms of Exner time (Texner between runs)
    
    # Flume geometry parameters
    W = run_param[0,4] # Flume width [m]
    S = run_param[0,5] # Flume slope
    
    # Sediment parameters
    d50 = run_param[0,6]

    # Run discharge
    Q = run_param[0,0] # Run discharge [l/s]
    
    # Create the list of surveys files and sore them in the files list.
    # To do so, two system array will be created for the run with more than 10 surveys due to the behaviour of the sort python function that sort files in this way:
    # matrix_bed_norm_q10_3s0.txt matrix_bed_norm_q10_3s1.txt matrix_bed_norm_q10_3s11.txt matrix_bed_norm_q10_3s12.txt matrix_bed_norm_q10_3s2.txt
    files = []
    files1=[] # Files list with surveys from 0 to 9
    files2=[] # Files list with surveys from 10
    # Creating array with file names:
    for f in sorted(os.listdir(input_dir)):
        path = os.path.join(input_dir, f)
        if os.path.isfile(path) and f.endswith('.txt') and f.startswith('matrix_bed_norm_'+run+'s'):
            files = np.append(files, f)
    for f in files:
        if len(f)==len(files[1]):
            files1 = np.append(files1, f)
        elif len(f)==len(files[1])+1:
            files2 = np.append(files2, f)
            
    files = np.append(files1,files2) # Files has been overwritten with a list of file names sorted in the right way :) 

    if process_mode==1:
        pass
    elif process_mode == 2:
        files=[]
        files=np.append(files,(DEM1_single_name, DEM2_single_name))

    # INITIALIZE ARRAYS
    comb = np.array([]) # combination of differences
    DoD_act_px_count_array=[] # Array where collect all the DoDs active pixel counting
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
    morphWact_values=[] # morphWact values for each section of all the DoD
    report_matrix = [] #Report matrix
    # matrix_volumes=np.zeros((len(files)-1, len(files)+1)) # Volumes report matrix
    matrix_volumes=np.zeros((len(files)-1, len(files)+1)) # Volumes report matrix
    matrix_sum_volumes=np.zeros((len(files)-1, len(files)+1)) # Sum of scour and deposition volumes
    # matrix_dep=np.zeros((len(files)-1, len(files)+1)) # Deposition volume report matrix
    matrix_dep=np.zeros((len(files)+3, len(files)+1)) # Deposition volume report matrix
    matrix_morph_act_area=np.zeros((len(files)+3, len(files)+1)) # Total active area report matrix
    matrix_morph_act_area_dep=np.zeros((len(files)+3, len(files)+1)) # Deposition active area report matrix
    matrix_morph_act_area_sco=np.zeros((len(files)+3, len(files)+1)) # Scour active area report matrix
    # matrix_sco=np.zeros((len(files)-1, len(files)+1)) # Scour volume report matrix
    matrix_sco=np.zeros((len(files)+3, len(files)+1)) # Scour volume report matrix
    matrix_Wact=np.zeros((len(files)+3, len(files)+3)) # Active width report matrix
    matrix_Wact_IIIquantile=np.zeros((len(files)-1, len(files)+1)) # III quantile active width report matrix
    matrix_Wact_Iquantile=np.zeros((len(files)-1, len(files)+1)) # I quantile active width report matrix
    matrix_act_thickness = np.zeros((len(files)-1, len(files)+1)) # Matrix where collect total active thickness data
    matrix_act_thickness_dep = np.zeros((len(files)-1, len(files)+1)) # Matrix where collect deposition active thickness data
    matrix_act_thickness_sco = np.zeros((len(files)-1, len(files)+1)) # Matrix where collect scour active thickness data
    matrix_act_volume = np.zeros((len(files)-1, len(files)+1)) # Matrix where collect volume data

    matrix_DEM_analysis = np.zeros((len(files), len(files)))

    ###########################################################################
    # CHECK DEMs SHAPE
    ###########################################################################
    # Due to differences between DEMs shape (not the same ScanArea.txt laser survey file)
    # a preliminary loop over the all DEMs is required in order to define the target
    # dimension of the reshaping operation
    array_dim_x = []
    array_dim_y = []
    for f in files:
        path_DEM = os.path.join(input_dir, f)
        DEM = np.loadtxt(path_DEM,
                          # delimiter=',',
                          skiprows=8
                          )
        array_dim_x = np.append(array_dim_x, DEM.shape[0])
        array_dim_y = np.append(array_dim_y, DEM.shape[1])

    # Define target dimension:
    shp_target_x, shp_target_y = int(min(array_dim_x)), int(min(array_dim_y))

    arr_shape = np.array([shp_target_x, shp_target_y]) # Define target shape


    ###########################################################################
    # SETUP MASKS
    ###########################################################################
    # array mask for filtering data outside the channel domain
    # Different mask will be applied depending on the run due to different ScanArea
    # used during the laser surveys
    runs_list = ['q10_1', 'q10_2', 'q15_1', 'q20_1', 'q20_2'] # Old runs with old ScanArea
    array_mask_name, array_mask_path = 'array_mask.txt', home_dir # Mask for runs 07 onwards

    if run in runs_list:
        array_mask_name, array_mask_path = 'array_mask_0.txt', home_dir
        print(array_mask_name)


    # Load mask
    array_mask = np.loadtxt(os.path.join(array_mask_path, array_mask_name))
    # Reshape mask:
    array_mask_rshp = array_mask[:shp_target_x,:shp_target_y] # Array mask reshaped

    # Create array mask:
    # - array_mask: np.array with 0 and 1
    # - array_mask_nan: np.array with np.nan and 1
    array_mask_rshp = np.where(array_mask_rshp==NaN, 0, 1) # Convert in mask with 0 and 1
    array_mask_rshp_nan = np.where(array_mask_rshp==0, np.nan, 1) # Convert in mask with np.nan and 1

    # Here we can split in two parts the DEMs or keep the entire one
    if mask_mode==1:
        pass
    elif mask_mode==2: # Working downstream, masking upstream
       array_mask_rshp[:,:-int(array_mask_rshp.shape[1]/2)] = NaN
       array_mask_rshp=np.where(array_mask_rshp==NaN, np.nan, array_mask_rshp)

    elif mask_mode==3: # Working upstream, masking downstream
        array_mask_rshp[:,int(array_mask_rshp.shape[1]/2):] = NaN
        array_mask_rshp=np.where(array_mask_rshp==NaN, np.nan, array_mask_rshp)

    ###########################################################################
    # LOOP OVER ALL DEMs COMBINATIONS
    ###########################################################################
    # Perform difference between DEMs over all the possible combination of surveys in the survey directory
    for h in range (0, len(files)-1):
        for k in range (0, len(files)-1-h):
            DEM1_name=files[h] # Extract the DEM1 name...
            DEM2_name=files[h+1+k] #...and the DEM2 name
            comb = np.append(comb, DEM2_name + '-' + DEM1_name) # Create a list with all the available combinations of DEMs

            # Overwrite DEM1 and DEM2 names in case of single DoD analysis
            if process_mode==1:
                pass
            elif process_mode==2:
                DEM1_name = DEM1_single_name
                DEM2_name = DEM2_single_name

            # Create DEMs paths...
            path_DEM1 = os.path.join(input_dir, DEM1_name)
            path_DEM2 = os.path.join(input_dir, DEM2_name)
            
            # ...and DOD name. The DoD name extraction depends by the length of
            # the DoD name sice for runs with more than 10 surveys the DEM's name is larger  
            if len(DEM1_name)==int(len(files[0])):
                DEM1_num = DEM1_name[-5:-4]
            elif len(DEM1_name)==int(len(files[0])+1):
                DEM1_num = DEM1_name[-6:-4]
                
            if len(DEM2_name)==int(len(files[0])):
                DEM2_num = DEM2_name[-5:-4]
            elif len(DEM2_name)==int(len(files[0])+1):
                DEM2_num = DEM2_name[-6:-4]
                
            DoD_name = 'DoD_' + DEM2_num + '-' + DEM1_num + '_'
            
            print('=========')
            print(DoD_name[:-1])
            print('=========')
            
            # # Setup output folder
            # output_name = 'script_outputs_' + DEM2_DoD_name + '-' + DEM1_DoD_name # Set outputs name

            # Set DoD outputs directory where to save DoD as ASCII grid and numpy matrix
            path_out = os.path.join(home_dir, 'DoDs', 'DoD_'+run)
            if not(os.path.exists(path_out)):
                os.mkdir(path_out)


            ###################################################################
            # DATA READING...
            ###################################################################
            # # Lines array and header array initialization and extraction:
            # lines = []
            # header = []

            # with open(path_DEM1, 'r') as file:
            #     for line in file:
            #         lines.append(line)  # lines is a list. Each item is a row of the input file
            #     # Header extraction...
            #     for i in range(0, 7):
            #         header.append(lines[i])
                    
            # # Header printing in a file txt called header.txt
            # with open(path_out + '/' + DoD_name + 'header.txt', 'w') as head:
            #     head.writelines(header)

            ###################################################################
            # DATA LOADING...
            ###################################################################
            # Load DEMs
            DEM1 = np.loadtxt(path_DEM1,
                              # delimiter=',',
                              skiprows=8
                              )
            DEM2 = np.loadtxt(path_DEM2,
                              # delimiter=',',
                              skiprows=8)


            # DEMs reshaping according to arr_shape...
            DEM1=DEM1[0:arr_shape[0], 0:arr_shape[1]]
            DEM2=DEM2[0:arr_shape[0], 0:arr_shape[1]]
            
            # Raster dimension
            dim_y, dim_x = DEM1.shape
            print('dim_x: ', dim_x, '    dim_y: ', dim_y)
            
            ###################################################################
            # HEADER
            ###################################################################
            # Lines array and header array initialization and extraction:
            lines = []
            header = []

            with open(path_DEM1, 'r') as file:
                for line in file:
                    lines.append(line)  # lines is a list. Each item is a row of the input file
                # Header extraction...
                for i in range(0, 7):
                    header.append(lines[i])
            
            # Update header columns and row number:
            header[4] = header[4].replace(header[4][22:25], str(dim_y))
            header[5] = header[5].replace(header[5][22:25], str(dim_x))
            
            # Header printing in a file txt called header.txt
            with open(path_out + '/' + DoD_name + 'header.txt', 'w') as head:
                head.writelines(header)
            
            ###################################################################
            # PERFORM DEM OF DIFFERENCE - DEM2-DEM1
            ###################################################################
            # Print DoD name
            print(DEM2_name, '-', DEM1_name)

            # Calculate the DoD length in meters:
            DoD_length = DEM1.shape[1]*px_x/1000 # DoD length [m]
            
            # DoD CREATION:
            # Creating DoD array with np.nan instead of NaN
            DoD_raw = np.zeros(DEM1.shape)
            DoD_raw = np.where(np.logical_or(DEM1 == NaN, DEM2 == NaN), np.nan, DEM2 - DEM1)
            
            # Masking with array mask
            DoD_raw = DoD_raw*array_mask_rshp_nan
            
            # Creating GIS readable DoD array (np.nan as -999)
            DoD_raw_rst = np.zeros(DoD_raw.shape)
            DoD_raw_rst = np.where(np.isnan(DoD_raw), NaN, DoD_raw)


            # Count the number of pixels in the channel area
            DoD_count = np.count_nonzero(np.where(np.isnan(DoD_raw), 0, 1))
            print('Number of channel pixel pixels:', DoD_count)
            
            # Append for each DoD the number of active pixels to the DoD_act_px_count_array
            DoD_act_px_count_array = np.append(DoD_act_px_count_array, DoD_count)

            # DoD statistics
            # print('The minimum DoD value is:\n', np.nanmin(DoD_raw))
            # print('The maximum DoD value is:\n', np.nanmax(DoD_raw))
            # print('The DoD shape is:\n', DoD_raw.shape)

            ###################################################################
            # DATA FILTERING...
            ###################################################################
            
            '''
            Output files:
                DoD_raw: it's just the dem of difference,  DEM2-DEM1
                DoD_raw_gis: the same for DoD_raw, but np.nan=NaN
                
                DoD_filt_mean: DoD_raw with a smoothing along the Y axes, see
                    the weight in the aspatial_weighted_average function
                DoD_filt_mean_gis: the same for DoD_filt_mean but np.nan=NaN
                
                DoD_filt_isol = DoD_filt_mean were the isolated_killer function
                    was performed
                DoD_filt_isol_gis = the same for DoD_filt_isol but np.nan=NaN
                
                DoD_filt_nature: DoD_filt_isol where the nature_checker function
                    was applied
                DoD_filt_nature_gis: the same for DoD_filt_nature but np.nan=NaN
                
                DoD_filt_fill: DoD_filt_nature where the isolated_filler
                    function was applied 
                DoD_filt_fill_gis: the same for DoD_filt_fill but np.nan=NaN
                
                DoD_filt_isol2: DoD_filt_fill where another round of isolated_killer
                    function was applied
                DoD_filt_isol2_gis: the same for DoD_filt_ult but with np.nan=NaN
                
                DoD_filt_ult: DoD_filt_isol2 where island_destryer function
                    was applied
                DoD_filt_ult_gis: the same for DoD_filt_ult but with np.nan=NaN
            '''
            
            # PERFORM DOMAIN-WIDE WEIGHTED AVERAGE:
            # -------------------------------------
            DoD_filt_mean, DoD_filt_mean_gis = spatial_weighted_average(DoD_raw, 1, NaN)
            
            # PERFORM UNDER THRESHOLD ZEROING:
            DoD_filt_mean = np.where(np.abs(DoD_filt_mean)<=thrs_1, 0, DoD_filt_mean)
            
            # PERFORM AVOIDING ZERO-SURROUNDED PIXEL PROCEDURE:
            #--------------------------------------------------
            # After trimming al np.nan values, counter represent the number of
            # pixel not equal to zero in the DoD_filt_mean matrix
            counter0 = np.count_nonzero(DoD_filt_mean[np.logical_not(np.isnan(DoD_filt_mean))])
            
            # Perform the very first isolated_killer procedure
            DoD_filt_isol, DoD_filt_isol_gis = isolated_killer(DoD_filt_mean, thrs_zeros, 1, NaN)
            
            # After trimming al np.nan values, counter represent the number of
            # pixel not equal to zero of the DoD_filt_isol matrix
            counter1 = np.count_nonzero(DoD_filt_isol[np.logical_not(np.isnan(DoD_filt_isol))])
            # Perform the isolated_killer procedure until the number of active
            # pixel will not change anymore 
            while counter0-counter1!=0:
                # Filtering...
                DoD_filt_isol, DoD_filt_isol_gis = isolated_killer(DoD_filt_isol, thrs_zeros, 1, NaN)
                # Update counters:
                counter0 = counter1
                counter1 = np.count_nonzero(DoD_filt_isol[np.logical_not(np.isnan(DoD_filt_isol))])
                
            # PERFORM NATURE CHECKER PIXEL PROCEDURE:
            #----------------------------------------
            DoD_filt_nature, DoD_filt_nature_gis = nature_checker(DoD_filt_isol, thrs_nature, 1, NaN)
            
            # PERFORM PITTS FILLING PIXEL PROCEDURE:
            #---------------------------------------
            # Initialize the counter from the previous step of the filtering process
            counter0 = np.count_nonzero(DoD_filt_nature[np.logical_not(np.isnan(DoD_filt_nature))])
            # Perform the first step of the filling procedure
            DoD_filt_fill, DoD_filt_fill_gis = isolated_filler(DoD_filt_nature, thrs_fill, 1, NaN)
            # Calculate the current counter
            counter1 = np.count_nonzero(DoD_filt_fill[np.logical_not(np.isnan(DoD_filt_fill))])
            # Perform the loop of the filtering process
            while counter0-counter1!=0:
                # Filtering...
                DoD_filt_fill, DoD_filt_fill_gis = isolated_filler(DoD_filt_fill, thrs_fill, 1, NaN)
                # Update counters:
                counter0=counter1
                counter1 = np.count_nonzero(DoD_filt_fill[np.logical_not(np.isnan(DoD_filt_fill))])
            
            # RE-PERFORM AVOIDING ZERO-SURROUNDED PIXEL PROCEDURE:
            #-----------------------------------------------------
            # After trimming al np.nan values, counter represent the number of
            # pixel not equal to zero o fthe DoD_filt_fill matrix
            counter0 = np.count_nonzero(DoD_filt_fill[np.logical_not(np.isnan(DoD_filt_fill))])
            
            # Perform the very first isolated_killer procedure
            DoD_filt_isol2, DoD_filt_isol2_gis = isolated_killer(DoD_filt_fill, thrs_zeros, 1, NaN)
            
            # After trimming al np.nan values, counter represent the number of
            # pixel not equal to zero of the DoD_filt_isol2 matrix
            counter1 = np.count_nonzero(DoD_filt_isol2[np.logical_not(np.isnan(DoD_filt_isol2))])
            # Perform the isolated_killer procedure until the number of active
            # pixel will not change anymore

            while counter0-counter1!=0:
                # Filtering...
                DoD_filt_isol2, DoD_filt_isol2_gis = isolated_killer(DoD_filt_isol2, thrs_zeros, 1, NaN)
                # Update counters:
                counter0 = counter1
                counter1 = np.count_nonzero(DoD_filt_isol2[np.logical_not(np.isnan(DoD_filt_isol2))])
            
            # PERFORM ISLAND DESTROYER PIXEL PROCEDURE:
            #------------------------------------------
            DoD_filt_ult, DoD_filt_ult_gis = island_destroyer(DoD_filt_isol2, 8, 1, NaN) # First step of filtering process
            
            for w in range(3,13): # Repeat the filtering process with windows from dimension 2 to 12
                DoD_filt_ult, DoD_filt_ult_gis = island_destroyer(DoD_filt_ult, w, 1, NaN)

            
            ###################################################################
            # PLOT RAW DOD, MEAN DOD AND FILTERED DOD
            ###################################################################
            # Print the last DoD outcome
            if save_plot_mode == 1:
                fig, ax = plt.subplots(dpi=200, tight_layout=True)
                # im = ax.imshow(np.where(DoD_filt_isol2_gis==NaN, np.nan, DoD_filt_ult_gis), cmap='RdBu',  vmin=-25, vmax=25, aspect='0.1')
                im = ax.imshow(DoD_filt_ult, cmap='RdBu',  vmin=-25, vmax=25, aspect='0.1')
                # plt.colorbar(im)
                plt.title(DoD_name[:-1], fontweight='bold')
                plt.savefig(os.path.join(plot_dir, run +'_DoD.png'), dpi=1600)
                plt.show()
            else:
                pass
            
            
            # PLOT OF ALL THE DIFFERENT FILTERING STAGE
            fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7) = plt.subplots(7,1, tight_layout=True, figsize=(10,6))
            fig.suptitle('Filtering process - ' + run)
            # Convert all zero value in np.nan to make it transparent on plots:
            DoD_raw_plot = np.where(DoD_raw==0, np.nan, DoD_raw)
            DoD_filt_mean_plot = np.array(np.where(DoD_filt_mean==0, np.nan, DoD_filt_mean))
            DoD_filt_isol_plot = np.array(np.where(DoD_filt_isol==0, np.nan, DoD_filt_isol))
            DoD_filt_nature_plot = np.array(np.where(DoD_filt_nature==0, np.nan, DoD_filt_nature))
            DoD_filt_fill_plot = np.array(np.where(DoD_filt_fill==0, np.nan, DoD_filt_fill))
            DoD_filt_isol2_plot = np.array(np.where(DoD_filt_isol2==0, np.nan, DoD_filt_isol2))
            DoD_filt_ult_plot = np.array(np.where(DoD_filt_ult==0, np.nan, DoD_filt_ult))
            
            raw = ax1.imshow(DoD_raw_plot, cmap='RdBu', aspect='0.1')
            ax1.set_title('raw DoD')

            filt_mean = ax2.imshow(DoD_filt_mean_plot, cmap='RdBu', aspect='0.1')
            ax2.set_title('filt_mean')

            filt_isol = ax3.imshow(DoD_filt_isol_plot, cmap='RdBu', aspect='0.1')
            ax3.set_title('filt_isol')
            
            filt_nature = ax4.imshow(DoD_filt_nature_plot, cmap='RdBu', aspect='0.1')
            ax4.set_title('filt_nature')
            
            filt_fill = ax5.imshow(DoD_filt_fill_plot, cmap='RdBu', aspect='0.1')
            ax5.set_title('filt_fill')
            
            filt_isol2 = ax6.imshow(DoD_filt_isol2_plot, cmap='RdBu', aspect='0.1')
            ax6.set_title('filt_isol2')
            
            filt_ult = ax7.imshow(DoD_filt_ult_plot, cmap='RdBu', aspect='0.1')
            ax7.set_title('filt_ult')
            
            # fig.colorbar(DoD_filt_isol2_plot)
            plt.savefig(os.path.join(plot_dir, run +'_'+DoD_name[:-1]+'_filtmap.tif'), dpi=1000) # raster (png, jpg, rgb, tif), vector (pdf, eps), latex (pgf)
            plt.show()
            

            ###################################################################
            # TOTAL VOLUMES, DEPOSITION VOLUMES AND SCOUR VOLUMES
            ###################################################################
            # DoD filtered name: DoD_filt
            # Create new raster where apply volume calculation
            # DoD>0 --> Deposition, DoD<0 --> Scour
            # =+SUMIFS(A1:JS144, A1:JS144,">0")*5*50(LibreCalc function)
            
            # SCOUR AND DEPOSITION MATRIX, DEPOSITION ONLY MATRIX AND SCOUR ONLY MATRIX:
            DoD_vol = np.where(np.isnan(DoD_filt_ult), 0, DoD_filt_ult) # Total volume matrix
            dep_DoD = (DoD_vol>0)*DoD_vol # Deposition only matrix
            sco_DoD = (DoD_vol<0)*DoD_vol # Scour only matrix
            
            # ...as boolean active pixel matrix:
            act_px_matrix = np.where(DoD_filt_ult!=0, 1, 0)*np.where(np.isnan(DoD_filt_ult), np.nan, 1) # Active pixel matrix, both scour and deposition
            act_px_matrix_dep = np.where(DoD_filt_ult>0, 1, 0)*np.where(np.isnan(DoD_filt_ult), np.nan, 1) # Active deposition matrix 
            act_px_matrix_sco = np.where(DoD_filt_ult<0, 1, 0)*np.where(np.isnan(DoD_filt_ult), np.nan, 1) # Active scour matrix
            
            # GIS readable matrix where np.nan is NaN
            act_px_matrix_gis = np.where(np.isnan(act_px_matrix), NaN, act_px_matrix) # Active pixel matrix, both scour and deposition
            act_px_matrix_dep_gis = np.where(np.isnan(act_px_matrix_dep), NaN, act_px_matrix_dep) # Active deposition matrix 
            act_px_matrix_sco_gis = np.where(np.isnan(act_px_matrix_sco), NaN, act_px_matrix_sco) # Active scour matrix
            
            # MORPHOLOGICAL QUANTITIES:
            tot_vol, sum_vol, dep_vol, sco_vol, morph_act_area, morph_act_area_dep, morph_act_area_sco, act_width_mean, act_width_mean_dep, act_width_mean_sco, act_thickness, act_thickness_dep, act_thickness_sco, bri = morph_quantities(DoD_filt_ult)

            
            #Print results:
            print('Total volume:', "{:.1f}".format(tot_vol))
            print('Sum of deposition and scour volume:', "{:.1f}".format(sum_vol))
            print('Deposition volume:', "{:.1f}".format(dep_vol))
            print('Scour volume:', "{:.1f}".format(sco_vol))

            # Append values to output data array
            volumes_array = np.append(volumes_array, tot_vol)
            sum_array = np.append(sum_array, sum_vol)
            dep_array = np.append(dep_array, dep_vol)
            sco_array = np.append(sco_array, sco_vol)
            
            
            
            ###################################################################
            # Active_pixel analysis
            ###################################################################            
            morph_act_area_array = np.append(morph_act_area_array, morph_act_area) # For each DoD, append total active area data
            morph_act_area_array_dep = np.append(morph_act_area_array_dep, morph_act_area_dep) # For each DoD, append deposition active area data
            morph_act_area_array_sco = np.append(morph_act_area_array_sco, morph_act_area_sco) # For each DoD, append scour active area data
            
            act_width_mean_array = np.append(act_width_mean_array, act_width_mean) # For each DoD append total active width values
            act_width_mean_array_dep = np.append(act_width_mean_array_dep, act_width_mean_dep) # For each DoD append deposition active width values
            act_width_mean_array_sco = np.append(act_width_mean_array_sco, act_width_mean_sco) # For each DoD append scour active width values
            
            act_width_array = np.array([np.nansum(act_px_matrix, axis=0)]) # Array of the crosswise morphological total active width in number of active cells [-]
            act_width_array_dep = np.array([np.nansum(act_px_matrix_dep, axis=0)]) # Array of the crosswise morphological deposition active width in number of active cells [-]
            act_width_array_sco = np.array([np.nansum(act_px_matrix_sco, axis=0)]) # Array of the crosswise morphological scour active width in number of active cells [-]
            
            print('Active thickness [mm]:', act_thickness)
            print('Morphological active area (number of active cells): ', "{:.1f}".format(morph_act_area), '[-]')
            print('Morphological active width (mean):', "{:.3f}".format(act_width_mean/(dim_y)), '%')
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

            # TODO UPDATE
            delta=int(DEM2_num)-int(DEM1_num) # Calculate delta between DEM
            
            print('delta = ', delta)
            print('----------')
            print()
            print()
            
            # Build up morphWact/W array for the current run boxplot
            # This array contain all the morphWact/W values for all the run repetition in the same line
            # This array contain only adjacent DEMs DoD
            if delta==1:
                morphWact_values = np.append(morphWact_values, act_width_array)

            # Fill Scour, Deposition and morphWact/w matrix:
            if delta != 0:
                # Fill matrix with data
                matrix_volumes[delta-1,h]=tot_vol # Total volumes as the algebric sum of scour and deposition volumes [L]
                matrix_sum_volumes[delta-1,h]=sum_vol # Total volumes as the sum of scour and deposition volumes [L]
                matrix_dep[delta-1,h]=dep_vol # Deposition volume [L]
                matrix_sco[delta-1,h]=sco_vol # Scour volume [L]
                matrix_morph_act_area[delta-1,h]=morph_act_area # Total morphological active area in number of cells [-]
                matrix_morph_act_area_dep[delta-1,h]=morph_act_area_dep # Deposition morphological active area in number of cells [-]
                matrix_morph_act_area_sco[delta-1,h]=morph_act_area_sco # Scour morphological active area in number of cells [-]
                matrix_act_thickness[delta-1,h]=act_thickness # Active thickness data calculated from total volume matrix [L]
                matrix_act_thickness_dep[delta-1,h]=act_thickness_dep # Active thickness data calculated from deposition volume matrix [L]
                matrix_act_thickness_sco[delta-1,h]=act_thickness_sco # Active thickness data calculated from scour volume matrix [L]
                matrix_Wact[delta-1,h]=act_width_mean
                
                # Fill last two columns with AVERAGE of the corresponding row
                matrix_volumes[delta-1,-2]=np.average(matrix_volumes[delta-1,:len(files)-delta]) #Total volumes
                matrix_sum_volumes[delta-1,-2]=np.average(matrix_sum_volumes[delta-1,:len(files)-delta]) #Total sum volumes
                matrix_dep[delta-1,-2]=np.average(matrix_dep[delta-1,:len(files)-delta]) # Deposition volumes
                matrix_sco[delta-1,-2]=np.average(matrix_sco[delta-1,:len(files)-delta]) # Scour volumes
                matrix_morph_act_area[delta-1,-2]=np.average(matrix_morph_act_area[delta-1,:len(files)-delta]) # Morphological total active area
                matrix_morph_act_area_dep[delta-1,-2]=np.average(matrix_morph_act_area_dep[delta-1,:len(files)-delta]) # Morphological deposition active area
                matrix_morph_act_area_sco[delta-1,-2]=np.average(matrix_morph_act_area_sco[delta-1,:len(files)-delta]) # Morphological scour active area
                matrix_act_thickness[delta-1,-2]=np.average(matrix_act_thickness[delta-1,:len(files)-delta]) # Fill matrix with active thickness average calculated from total volume matrix
                matrix_act_thickness_dep[delta-1,-2]=np.average(matrix_act_thickness_dep[delta-1,:len(files)-delta]) # Active thickness average calculated from deposition volume matrix
                matrix_act_thickness_sco[delta-1,-2]=np.average(matrix_act_thickness_sco[delta-1,:len(files)-delta]) # Active thickness average calculated from scour volume matrix
                matrix_Wact[delta-1,-2]=np.average(matrix_Wact[delta-1,:len(files)-delta])
                
                # Fill last two columns with STDEV of the corresponding row
                matrix_volumes[delta-1,-1]=np.std(matrix_volumes[delta-1,:len(files)-delta])
                matrix_sum_volumes[delta-1,-1]=np.std(matrix_sum_volumes[delta-1,:len(files)-delta])
                matrix_dep[delta-1,-1]=np.std(matrix_dep[delta-1,:len(files)-delta])
                matrix_sco[delta-1,-1]=np.std(matrix_sco[delta-1,:len(files)-delta])
                matrix_morph_act_area[delta-1,-1]=np.std(matrix_morph_act_area[delta-1,:len(files)-delta])
                matrix_morph_act_area_dep[delta-1,-1]=np.std(matrix_morph_act_area_dep[delta-1,:len(files)-delta])
                matrix_morph_act_area_sco[delta-1,-1]=np.std(matrix_morph_act_area_sco[delta-1,:len(files)-delta])
                matrix_act_thickness[delta-1,-1]=np.std(matrix_act_thickness[delta-1,:len(files)-delta]) # Fill matrix with active thickness standard deviation calculated from total volume matrix
                matrix_act_thickness_dep[delta-1,-1]=np.std(matrix_act_thickness_dep[delta-1,:len(files)-delta]) # Active thickness average calculated from deposition volume matrix
                matrix_act_thickness_sco[delta-1,-1]=np.std(matrix_act_thickness_sco[delta-1,:len(files)-delta]) # Active thickness average calculated from scour volume matrix
                matrix_Wact[delta-1,-1]=np.std(matrix_Wact[delta-1,:len(files)-delta])
                
                # Fill III quantile Wact/W matrix:
                matrix_Wact_IIIquantile[delta-1,h]=np.quantile(act_width_array, .75)
                matrix_Wact_IIIquantile[delta-1,-2]=np.min(matrix_Wact_IIIquantile[delta-1,:len(files)-delta])
                matrix_Wact_IIIquantile[delta-1,-1]=np.max(matrix_Wact_IIIquantile[delta-1,:len(files)-delta])

                # Fill I quantile Wact/W matrix:
                matrix_Wact_Iquantile[delta-1,h]=np.quantile(act_width_array, .25)
                matrix_Wact_Iquantile[delta-1,-2]=np.min(matrix_Wact_Iquantile[delta-1,:len(files)-delta])
                matrix_Wact_Iquantile[delta-1,-1]=np.max(matrix_Wact_Iquantile[delta-1,:len(files)-delta])   
                

                # DATA STRUCTURE
                # Fill Wact/W MEAN matrix as below:
                # DoD step0  1-0   2-1   3-2   4-3   5-4   6-5   7-6   8-7   9-8  Iquantile IIIquantile average STDEV
                # DoD step1  2-0   3-1   4-2   5-3   6-4   7-5   8-6   9-7        Iquantile IIIquantile average STDEV
                # DoD step2  3-0   4-1   5-2   6-3   7-4   8-5   9-6              Iquantile IIIquantile average STDEV
                # DoD step3  4-0   5-1   6-2   7-3   8-4   9-5                    Iquantile IIIquantile average STDEV
                # DoD step4  5-0   6-1   7-2   8-3   9-4                          Iquantile IIIquantile average STDEV
                # DoD step5  6-0   7-1   8-2   9-3                                Iquantile IIIquantile average STDEV
                # DoD step6  7-0   8-1   9-2                                      Iquantile IIIquantile average STDEV
                # DoD step7  8-0   9-1                                            Iquantile IIIquantile average STDEV
                # DoD step8  9-0                                                  Iquantile IIIquantile average STDEV

                # Fill Wact/W MAX (MIN) matrix as below:
                # NB: MIN and MAX columns are to be intended as the maximum and the minimum value
                # of the IIIquantile (or Iquantile) values of DoDs row. So the MIN value of the
                # matrix_Wact_IIIquantile is the minimum value between the maximum value.
                # DoD step0  1-0   2-1   3-2   4-3   5-4   6-5   7-6   8-7   9-8  min(Iquantile) max(Iquantile)
                # DoD step1  2-0   3-1   4-2   5-3   6-4   7-5   8-6   9-7        min(Iquantile) max(Iquantile)
                # DoD step2  3-0   4-1   5-2   6-3   7-4   8-5   9-6              min(Iquantile) max(Iquantile)
                # DoD step3  4-0   5-1   6-2   7-3   8-4   9-5                    min(Iquantile) max(Iquantile)
                # DoD step4  5-0   6-1   7-2   8-3   9-4                          min(Iquantile) max(Iquantile)
                # DoD step5  6-0   7-1   8-2   9-3                                min(Iquantile) max(Iquantile)
                # DoD step6  7-0   8-1   9-2                                      min(Iquantile) max(Iquantile)
                # DoD step7  8-0   9-1                                            min(Iquantile) max(Iquantile)
                # DoD step8  9-0                                                  min(Iquantile) max(Iquantile)
                
                
            else:
                pass


            ###################################################################
            # STACK CONSECUTIVE DoDS IN A 3D ARRAY
            ###################################################################
            # Initialize 3D array to stack DoDs
            if h==0 and k==0: # initialize the first array with the DEM shape
                DoD_stack1 = np.zeros([len(files)-1, dim_y, dim_x])
            else:
                pass
            # Stack all the DoDs inside the 3D array
            if delta==1:
                DoD_stack1[h,:,:] = DoD_filt_ult_gis[:,:]

            ###################################################################
            # DoDs SAVING...
            ###################################################################

            os.path.join(path_out, DoD_name, )
            
            # RAW DoD
            # Print raw DoD in txt file (NaN as np.nan)
            np.savetxt(os.path.join(path_out, DoD_name + 'raw.txt'), DoD_raw, fmt='%0.1f', delimiter='\t')
            # Printing raw DoD in txt file (NaN as -999)
            np.savetxt(os.path.join(path_out, DoD_name + 'raw_gis.txt'), DoD_raw_rst, fmt='%0.1f', delimiter='\t')

            # WEIGHTED AVERAGED DoD
            # Print DoD mean in txt file (NaN as np.nan)
            np.savetxt(os.path.join(path_out, DoD_name + 'filt_mean.txt'), DoD_filt_mean , fmt='%0.1f', delimiter='\t')
            # Print filtered DoD (with NaN as -999)
            np.savetxt(os.path.join(path_out, DoD_name + 'filt_mean_gis.txt'), DoD_filt_mean_gis , fmt='%0.1f', delimiter='\t')

            # ISOLATE KILLER FUNCTION APPLIED DoD
            # Print filtered DoD (with np.nan)...
            np.savetxt(os.path.join(path_out, DoD_name + 'filt_isol.txt'), DoD_filt_isol, fmt='%0.1f', delimiter='\t')
            # Print filtered DoD (with NaN as -999)
            np.savetxt(os.path.join(path_out, DoD_name + 'filt_isol_gis.txt'), DoD_filt_isol_gis, fmt='%0.1f', delimiter='\t')
            
            # NATURE CHECKER FUNCTION APPLIED DoD
            # Print filtered DoD (with np.nan)...
            np.savetxt(os.path.join(path_out, DoD_name + 'filt_nature.txt'), DoD_filt_nature, fmt='%0.1f', delimiter='\t')
            # Print filtered DoD (with NaN as -999)
            np.savetxt(os.path.join(path_out, DoD_name + 'filt_nature_gis.txt'), DoD_filt_nature_gis, fmt='%0.1f', delimiter='\t')
            
            # ISOLATE FILLER FUNCTION APPLIED DoD
            # Print filtered DoD (with np.nan)...
            np.savetxt(os.path.join(path_out, DoD_name + 'filt_fill.txt'), DoD_filt_fill, fmt='%0.1f', delimiter='\t')
            # Print filtered DoD (with NaN as -999)
            np.savetxt(os.path.join(path_out, DoD_name + 'filt_fill_gis.txt'), DoD_filt_fill_gis, fmt='%0.1f', delimiter='\t')

            # SECOND ROUND OF ISOLATE KILLER FUNCTION APPLIED DoD (This is the ultimate DoD)
            # Print filtered DoD (with np.nan)...
            np.savetxt(os.path.join(path_out, DoD_name + 'filt_isol2.txt'), DoD_filt_isol2, fmt='%0.1f', delimiter='\t')
            # Print filtered DoD (with NaN as -999)
            np.savetxt(os.path.join(path_out, DoD_name + 'filt_isol2_gis.txt'), DoD_filt_isol2_gis, fmt='%0.1f', delimiter='\t')
            
            # ISLAND KILLER FUNCTION APPLIED DoD (...)
            # Print filtered DoD (with np.nan)...
            np.savetxt(os.path.join(path_out, DoD_name + 'filt_ult.txt'), DoD_filt_ult, fmt='%0.1f', delimiter='\t')
            # Print filtered DoD (with NaN as -999)
            np.savetxt(os.path.join(path_out, DoD_name + 'filt_ult_gis.txt'), DoD_filt_ult_gis, fmt='%0.1f', delimiter='\t')

            # ACTIVE PIXEL DoD
            # Print boolean map of active pixel: 1=active, 0=not active
            np.savetxt(os.path.join(path_out, DoD_name + 'activity_map.txt'), act_px_matrix, fmt='%0.1f', delimiter='\t')
            # Print boolean GIS readable map of active pixel as above (np.nan is NaN)
            np.savetxt(os.path.join(path_out, DoD_name + 'activity_map_gis.txt'), act_px_matrix_gis , fmt='%0.1f', delimiter='\t')
            
            # ACTIVE DEPOSITION PIXEL DoD
            # Print boolean map of active pixel: 1=deposition, 0=not active or scour
            np.savetxt(os.path.join(path_out, DoD_name + 'activity_map_dep.txt'), act_px_matrix_dep, fmt='%0.1f', delimiter='\t')
            # Print boolean GIS readable map of active pixel as above (np.nan is NaN)
            np.savetxt(os.path.join(path_out, DoD_name + 'activity_map_dep_gis.txt'), act_px_matrix_dep_gis , fmt='%0.1f', delimiter='\t')
            
            # ACTIVE SCOUR PIXEL DoD
            # Print boolean map of active pixel: 1=scour, 0=not active or deposition
            np.savetxt(os.path.join(path_out, DoD_name + 'activity_map_sco.txt'), act_px_matrix_sco, fmt='%0.1f', delimiter='\t')
            # Print boolean GIS readable map of active pixel as above (np.nan is NaN)
            np.savetxt(os.path.join(path_out, DoD_name + 'activity_map_sco_gis.txt'), act_px_matrix_sco_gis, fmt='%0.1f', delimiter='\t')


            # TODO Could this part be implemented as a function?
            # Print DoD and filtered DoD (with NaN as -999) in a GIS readable format (ASCII grid):
            with open(os.path.join(path_out, DoD_name + 'header.txt')) as f_head:
                w_header = f_head.read()    # Header
            with open(os.path.join(path_out, DoD_name + 'raw_gis.txt')) as f_DoD:
                w_DoD_raw= f_DoD.read()
            with open(os.path.join(path_out, DoD_name + 'filt_mean_gis.txt')) as f_DoD_mean:
                w_DoD_mean = f_DoD_mean.read()
            with open(os.path.join(path_out, DoD_name + 'filt_isol_gis.txt')) as f_DoD_isol:
                w_DoD_isol = f_DoD_isol.read()
            with open(os.path.join(path_out, DoD_name + 'filt_nature_gis.txt')) as f_DoD_nature:
                w_DoD_nature = f_DoD_nature.read()
            with open(os.path.join(path_out, DoD_name + 'filt_fill_gis.txt')) as f_DoD_fill:
                w_DoD_fill = f_DoD_fill.read()
            with open(os.path.join(path_out, DoD_name + 'filt_isol2_gis.txt')) as f_DoD_isol2:
                w_DoD_isol2 = f_DoD_isol2.read()
            with open(os.path.join(path_out, DoD_name + 'filt_ult_gis.txt')) as f_DoD_ult:
                w_DoD_ult = f_DoD_ult.read()
            with open(os.path.join(path_out, DoD_name + 'activity_map_gis.txt')) as f_DoD_act_map:
                w_DoD_act_map = f_DoD_act_map.read()
            with open(os.path.join(path_out, DoD_name + 'activity_map_dep_gis.txt')) as f_DoD_act_map_dep:
                w_DoD_act_map_dep = f_DoD_act_map_dep.read()
            with open(os.path.join(path_out, DoD_name + 'activity_map_sco_gis.txt')) as f_DoD_act_map_sco:
                w_DoD_act_map_sco = f_DoD_act_map_sco.read()

                # Print GIS readable raster
                DoD_raw_gis = w_header + w_DoD_raw
                DoD_mean_gis = w_header + w_DoD_mean
                DoD_isol_gis = w_header + w_DoD_isol
                DoD_nature_gis = w_header + w_DoD_nature
                DoD_fill_gis = w_header + w_DoD_fill
                DoD_isol2_gis = w_header + w_DoD_isol2
                DoD_ult_gis = w_header + w_DoD_ult
                DoD_act_map_gis = w_header + w_DoD_act_map
                DoD_act_map_dep_gis = w_header + w_DoD_act_map_dep
                DoD_act_map_sco_gis = w_header + w_DoD_act_map_sco

            with open(os.path.join(path_out, DoD_name + 'raw_gis.txt'), 'w') as fp:
                fp.write(DoD_raw_gis)
            with open(os.path.join(path_out, DoD_name + 'filt_mean_gis.txt'), 'w') as fp:
                fp.write(DoD_mean_gis)
            with open(os.path.join(path_out, DoD_name + 'filt_isol_gis.txt'), 'w') as fp:
                fp.write(DoD_isol_gis)
            with open(os.path.join(path_out, DoD_name + 'filt_nature_gis.txt'), 'w') as fp:
                fp.write(DoD_nature_gis)
            with open(os.path.join(path_out, DoD_name + 'filt_fill_gis.txt'), 'w') as fp:
                fp.write(DoD_fill_gis)
            with open(os.path.join(path_out, DoD_name + 'filt_isol2_gis.txt'), 'w') as fp:
                fp.write(DoD_isol2_gis)
            with open(os.path.join(path_out, DoD_name + 'filt_ult_gis.txt'), 'w') as fp:
                fp.write(DoD_ult_gis)
            with open(os.path.join(path_out, DoD_name + 'activity_map_gis.txt'), 'w') as fp:
                fp.write(DoD_act_map_gis)
            with open(os.path.join(path_out, DoD_name + 'activity_map_dep_gis.txt'), 'w') as fp:
                fp.write(DoD_act_map_dep_gis)
            with open(os.path.join(path_out, DoD_name + 'activity_map_sco_gis.txt'), 'w') as fp:
                fp.write(DoD_act_map_sco_gis)
    
    ###################################################################
    # DoDs STACK SAVING...
    ###################################################################
    # TODO go on with this section
    DoD_stack_nan1 = np.where(DoD_stack1 == NaN, np.nan, DoD_stack1)
    
    # Save 3D array as binary file
    np.save(os.path.join(DoDs_dir, 'DoDs_stack',"DoD_stack1_"+run+".npy"), DoD_stack_nan1)
    
    # Create 3D array where scours are -1, depositions are +1 and no changes are 0
    
    DoD_stack_bool1 = np.where(DoD_stack1>0, 1, DoD_stack_nan1)
    DoD_stack_bool1 = np.where(DoD_stack_nan1<0, -1, DoD_stack_bool1)
    
    # Save 3D array as binary file
    np.save(os.path.join(DoDs_dir, 'DoDs_stack',"DoD_stack_bool1_"+run+".npy"), DoD_stack_bool1)


    # Fill DoD lenght array
    DoD_length_array = np.append(DoD_length_array, DoD_length)



    # PRINT THE LAST DOD OUTCOME
    if save_plot_mode == 1:
        fig, ax = plt.subplots(dpi=200, tight_layout=True)
        # im = ax.imshow(np.where(DoD_filt_ult_gis==NaN, np.nan, DoD_filt_ult_gis), cmap='RdBu',  vmin=-25, vmax=25, aspect='0.1')
        im = ax.imshow(DoD_filt_ult, cmap='RdBu',  vmin=-25, vmax=25, aspect='0.1')
        # plt.colorbar(im)
        plt.title(DoD_name[:-1], fontweight='bold')
        plt.savefig(os.path.join(plot_dir, run +'_DoD.png'), dpi=1600)
        plt.show()
    else:
        pass
    


    ###########################################################################
    # VOLUME AND MORPHOLOGICA ACTIVE WIDTH INTERPOLATION
    ###########################################################################
    if data_interpolation_mode == 1:
        '''
        Interpolation performed all over the volume data.
        Standard deviation is then applied to function parameters
        '''
        # Initialize arrays
        xData=[] # xData as time array
        yData_dep=[] # yData_dep deposition volume array
        yData_sco=[] # yData_sco scour volume array
        yData_morphW=[] # yData_morphW morphological active width array
    
        for i in range(0,len(files)-1):
            xData=np.append(xData, np.ones(len(files)-i-1)*(i+1)*dt) # Create xData array for all the volume points
            yData_dep=np.append(yData_dep, matrix_dep[i,:len(files)-i-1]) # deposition volumes (unroll yData)
            yData_sco=np.append(yData_sco, abs(matrix_sco[i,:len(files)-i-1])) # scour volumes (unroll yData)
            yData_morphW=np.append(yData_morphW, abs(matrix_Wact[i,:len(files)-i-1])) # scour volumes (unroll yData)
    
    
    
        # Define interpolation array and initial guess:
        volume_temp_scale_array = [] # Define volume temporal scale array
        morphW_temp_scale_array = [] # Define morphW temporal scale array
        ic_dep=np.array([np.mean(yData_dep),np.min(xData)]) # Initial deposition parameter guess
        ic_sco=np.array([np.mean(yData_sco),np.min(xData)]) # Initial scour parameter guess
        ic_morphW=np.array([np.mean(yData_morphW),np.min(xData)]) # Initial morphW parameter guess
    
        # Perform interpolation for deposition and scour volumes, and for morphological active width
        par_dep, intCurve_dep, covar_dep = interpolate(func_exp, xData, yData_dep, ic_dep) # Deposition interpolation
        par_sco, intCurve_sco, covar_sco = interpolate(func_exp, xData, yData_sco, ic_sco) # Scour interpolation
        par_morphW, intCurve_morphW, covar_morphW = interpolate(func_exp3, xData, yData_morphW, ic_morphW) # morphW interpolation
    
    
       # Build up volume temporal scale array for each runs
        if run_mode==2:
            volume_temp_scale_array = np.append(volume_temp_scale_array, (par_dep[1], covar_dep[1,1], par_sco[1], covar_sco[1,1])) # Append values
            volume_temp_scale_report[int(np.where(RUNS==run)[0]),:]=volume_temp_scale_array # Populate temporal scale report
    
        # Build up morphW temporal scale array for each runs
        if run_mode==2:
            morphW_temp_scale_array = np.append(morphW_temp_scale_array, (par_morphW[1], covar_morphW[1,1])) # Append values
            morphW_temp_scale_report[int(np.where(RUNS==run)[0]),:]=morphW_temp_scale_array # Populate temporal scale report
    
        print()
        print('All volume points interpolation parameters:')
        print('Deposition interpolation parameters')
        print('A=', par_dep[0], 'Variance=', covar_dep[0,0])
        print('B=', par_dep[1], 'Variance=', covar_dep[1,1])
        print('Scour interpolation parameters')
        print('A=', par_sco[0], 'Variance=', covar_sco[0,0])
        print('B=', par_sco[1], 'Variance=', covar_sco[1,1])
        print()
        print('All morphW points interpolation parameters:')
        print('A=', par_morphW[0], 'Variance=', covar_morphW[0,0])
        print('B=', par_morphW[1], 'Variance=', covar_morphW[1,1])


        if save_plot_mode == 1:
            fig1, axs = plt.subplots(2,1,dpi=200, sharex=True, tight_layout=True)
            axs[0].plot(xData, yData_dep, 'o')
            axs[0].plot(xData, intCurve_dep, c='red')
            axs[0].set_title('Deposition volumes interpolation '+run)
            axs[0].set_xlabel('Time [min]')
            axs[0].set_ylabel('Volume V/(L*W) [mm]')
            axs[1].plot(xData, yData_sco, 'o')
            axs[1].plot(xData, intCurve_sco, c='red')
            axs[1].set_title('Scour volumes interpolation '+run)
            axs[1].set_xlabel('Time [min]')
            axs[1].set_ylabel('Volume V/(L*W) [mm]')
            plt.savefig(os.path.join(plot_dir, run +'_volume_interp.png'), dpi=200)
            plt.show()
    
            fig2, axs = plt.subplots(1,1,dpi=200, sharex=True, tight_layout=True)
            axs.plot(xData, yData_morphW, 'o', c='brown')
            axs.plot(xData, intCurve_morphW, c='green')
            axs.set_title('Morphological active width (morphW/W) '+run)
            axs.set_xlabel('Time [min]')
            axs.set_ylabel('morphW/W [-]')
            plt.savefig(os.path.join(plot_dir, run +'_morphW_interp.png'), dpi=200)
            plt.show()
    
        else:
            pass
    else:
        pass


    # # Fill scour and deposition report matrix with interpolation parameters
    # for i in range(0, len(files)-3): # Last three columns have 1 or 2 or 3 values: not enought -> interpolation skipped
    #     xData = np.ones(len(files)-i-1)*(i+1)*dt # Create xData array for all the volume points
    #     # xData = np.arange(0, len(files)-i-1, 1)

    #     #Fill deposition matrix
    #     yData_dep=matrix_dep[:len(files)-i-1,i] # yData as value of deposition volume
    #     ic_dep=np.array([np.mean(yData_dep),np.min(xData)]) # Initial deposition parameter guess
    #     par_dep, intCurve, covar_dep = interpolate(func, xData, yData_dep, ic_dep)
    #     matrix_dep[-4,i],  matrix_dep[-2,i]=  par_dep[0], par_dep[1] # Parameter A and B
    #     matrix_dep[-3,i],  matrix_dep[-1,i]=  covar_dep[0,0], covar_dep[1,1] # STD(A) and STD(B)

    #     # Fill scour matrix
    #     yData_sco=np.absolute(matrix_sco[:len(files)-i-1,i])
    #     ic_sco=np.array([np.mean(yData_sco),np.min(xData)]) # Initial scour parameter guess
    #     par_sco, intCurve, covar_sco = interpolate(func, xData, yData_sco, ic_sco)
    #     matrix_sco[-4,i],  matrix_sco[-2,i]=  par_sco[0], par_sco[1] # Parameter A and B
    #     matrix_sco[-3,i],  matrix_sco[-1,i]=  covar_sco[0,0], covar_sco[1,1] # STD(A) and STD(B)

    #     print(xData)
    #     print(yData_dep)
    #     print(yData_sco)

    ###############################################################################
    # SAVE DATA MATRIX
    ###############################################################################
    # Create report matrix
    report_matrix = np.array(np.transpose(np.stack((comb, DoD_act_px_count_array, volumes_array, dep_array, sco_array, morph_act_area_array, act_width_mean_array))))
    report_header = 'DoD_combination, Active pixels, Total volume [mm^3], Deposition volume [mm^3], Scour volume [mm^3], Active area [mm^2], Active width mean [%]'

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


    # Create total sum volumes matrix report
    # TODO
    report_sum_vol_name = os.path.join(report_dir, run +'_sum_vol_report.txt')
    np.savetxt(report_sum_vol_name, matrix_sum_volumes, fmt='%.3f', delimiter=',', newline='\n')
    
    # Create deposition matrix report
    report_dep_name = os.path.join(report_dir, run +'_dep_report.txt')
    np.savetxt(report_dep_name, matrix_dep, fmt='%.3f', delimiter=',', newline='\n')

    # Create scour matrix report
    report_sco_name = os.path.join(report_dir, run +'_sco_report.txt')
    np.savetxt(report_sco_name, matrix_sco, fmt='%.3f', delimiter=',', newline='\n')
    
    # Create total active thickness matrix report (calculated from volume matrix)
    report_act_thickness_name = os.path.join(report_dir, run +'_act_thickness_report.txt')
    np.savetxt(report_act_thickness_name, matrix_act_thickness , fmt='%.3f', delimiter=',', newline='\n')
    
    # Create deposition active thickness matrix report (calculated from deposition volume matrix)
    report_act_thickness_name_dep = os.path.join(report_dir, run +'_act_thickness_report_dep.txt')
    np.savetxt(report_act_thickness_name_dep, matrix_act_thickness_dep , fmt='%.3f', delimiter=',', newline='\n')
    
    # Create scour active thickness matrix report (calculated from scour volume matrix)
    report_act_thickness_name_sco = os.path.join(report_dir, run +'_act_thickness_report_sco.txt')
    np.savetxt(report_act_thickness_name_sco, matrix_act_thickness_sco , fmt='%.3f', delimiter=',', newline='\n')
    
    # Create total active area matrix report (calculated from volume matrix)
    report_act_area_name = os.path.join(report_dir, run + '_act_area_report.txt')
    np.savetxt(report_act_area_name, matrix_morph_act_area, fmt='%.3f', delimiter=',', newline='\n')
    
    # Create deposition active area matrix report (calculated from volume matrix)
    report_act_area_name_dep = os.path.join(report_dir, run + '_act_area_report_dep.txt')
    np.savetxt(report_act_area_name_dep, matrix_morph_act_area_dep, fmt='%.3f', delimiter=',', newline='\n')
    
    # Create scour active area matrix report (calculated from volume matrix)
    report_act_area_name_sco = os.path.join(report_dir, run + '_act_area_report_sco.txt')
    np.savetxt(report_act_area_name_sco, matrix_morph_act_area_sco, fmt='%.3f', delimiter=',', newline='\n')

    # Create Wact report matrix
    matrix_Wact=matrix_Wact[:len(files)-1,:] # Fill matrix_Wact with morphological  active width values
    matrix_Wact[:,len(files)-1]=matrix_Wact_Iquantile[:,len(files)-1] # Fill matrix_Wact report with minimum values
    matrix_Wact[:,len(files)]=matrix_Wact_IIIquantile[:,len(files)] # Fill matrix_Wact report with maximum values
    report_Wact_name = os.path.join(report_dir, run +'_morphWact_report.txt')
    np.savetxt(report_Wact_name, matrix_Wact, fmt='%.3f', delimiter=',', newline='\n')

    # For each runs collect the dimension of the morphWact_array:
    if delta==1:
        morphWact_dim = np.append(morphWact_dim, len(morphWact_values))


    # Create morphWact/W matrix as following:
    # all morphWact/W values are appended in the same line for each line in the morphWact_values array
    # Now a matrix in which all row are all morphWact/W values for each runs is built
    # morphWact_matrix_header = 'run name, morphWact/W [-]'
    # run name, morphWact/w [-]
    with open(os.path.join(report_dir, run + '_morphWact_array.txt'), 'w') as fp:
        # fp.write(morphWact_matrix_header)
        # fp.writelines(['\n'])
        for i in range(0, len(morphWact_values)):
            if i == len(morphWact_values)-1:
                fp.writelines(["%.3f" % float(morphWact_values[i])])
            else:
                fp.writelines(["%.3f," % float(morphWact_values[i])])
        fp.writelines(['\n'])
    fp.close()



    ###########################################################################
    # PLOTS
    ###########################################################################
    # Define arrays for scour and volume data over time
    xData1=np.arange(1, len(files), 1)*dt_xnr # Time in Txnr
    yData_sco=np.absolute(matrix_sco[:len(files)-1,0])
    yError_sco=matrix_sco[:len(files)-1,-1]
    yData_dep=np.absolute(matrix_dep[:len(files)-1,0])
    yError_dep=matrix_dep[:len(files)-1,-1]
    yData_act_thickness=matrix_act_thickness[:len(files)-1,0]
    yError_act_thickness=matrix_act_thickness[:len(files)-1,-1]

    if save_plot_mode==1:
        fig3, axs = plt.subplots(2,1,dpi=80, figsize=(10,6), sharex=True, tight_layout=True)
        fig3.suptitle(run + ' - Volume')
        axs[0].errorbar(xData1,yData_sco, yError_sco, linestyle='--', marker='^', color='red')
        axs[0].set_ylim(bottom=0)
        axs[0].set_title('Scour')
        # axs[0].set_xlabel()
        axs[0].set_ylabel('Scour volume V/(L*W*d50) [-]')
        axs[1].errorbar(xData1,yData_dep, yError_dep, linestyle='--', marker='^', color='blue')
        axs[1].set_ylim(bottom=0)
        axs[1].set_title('Deposition')
        axs[1].set_xlabel('Exner time')
        axs[1].set_ylabel('Scour olume V/(L*W*d50) [-]')
        plt.savefig(os.path.join(plot_dir, run +'dep_scour.png'), dpi=200)
        plt.show()
        
        
        fig4, axs = plt.subplots(1,1,dpi=80, figsize=(10,6), sharex=True, tight_layout=True)
        axs.errorbar(xData1,yData_act_thickness, yError_act_thickness, linestyle='--', marker='^', color='purple')
        axs.set_ylim(bottom=0)
        axs.set_title(run + '- Active thickness')
        axs.set_xlabel('Exner time')
        axs.set_ylabel('Active thickness [mm]')
        plt.savefig(os.path.join(plot_dir, run +'active_thickness_.png'), dpi=200)
        plt.show()
    else:
        pass

    # # Print a report with xData as real time in minutes and  the value of scour and deposition volumes for each runs
    # Create report matrix as:
    # run
    # time
    # V_dep
    # V_sco

    xData2=np.arange(1, len(files), 1)*dt
    volume_over_time_matrix = []
    volume_over_time_matrix = np.stack((xData2, yData_dep, -yData_sco))

    # Append rows to the current file
    with open(os.path.join(report_dir, 'volume_over_time.txt'), 'a') as fp:
        fp.writelines([run+', '])
        fp.writelines(['\n'])
        for i in range(0,volume_over_time_matrix.shape[0]):
            for j in range(0,volume_over_time_matrix.shape[1]):
                fp.writelines(["%.3f, " % float(volume_over_time_matrix[i,j])])
            fp.writelines(['\n'])
        fp.writelines(['\n'])
    fp.close()
    
DoD_length_array = np.append(DoD_length_array, DoD_length)





    # if save_plot_mode == 1:
    #     # Print scour volumes over increasing timestep:
    #     fig1, ax1 = plt.subplots(dpi=100)
    #     # ax1.bar(np.arange(0, len(matrix_sco[:,0]), 1),abs(matrix_sco[:,0]))
    #     # ax1.plot(t[int(len(t)/10):-int(len(t)/10)], m*t[int(len(t)/10):-int(len(t)/10)]+q)
    #     # xData=np.arange(1, len(files), 1)*dt # Time in minutes
    #     xData=np.arange(1, len(files), 1)*dt_xnr # Time in Txnr
    #     yData=np.absolute(matrix_sco[:len(files)-1,0])
    #     yError=matrix_sco[:len(files)-1,-1]
    #     ax1.errorbar(xData,yData, yError, linestyle='--', marker='^')
    #     ax1.set_ylim(bottom=0)
    #     ax1.set_title(run)
    #     ax1.set_xlabel('Exner time')
    #     ax1.set_ylabel('Scour volume [mm³]')
    #     plt.savefig(os.path.join(plot_dir, run +'_scour.png'), dpi=200)
    #     plt.show()

    #     # Print deposition volumes over increasing timestep:
    #     fig1, ax1 = plt.subplots(dpi=100)
    #     # ax1.bar(np.arange(0, len(matrix_sco[:,0]), 1),abs(matrix_sco[:,0]))
    #     # ax1.plot(t[int(len(t)/10):-int(len(t)/10)], m*t[int(len(t)/10):-int(len(t)/10)]+q)
    #     # xData=np.arange(1, len(files), 1)*dt # Time in minutes
    #     xData=np.arange(1, len(files), 1)*dt_xnr # Time in Txnr
    #     yData=np.absolute(matrix_dep[:len(files)-1,0])
    #     yError=matrix_sco[:len(files)-1,-1]
    #     ax1.errorbar(xData,yData, yError, linestyle='--', marker='^')
    #     ax1.set_ylim(bottom=0)
    #     ax1.set_title(run)
    #     ax1.set_xlabel('Exner time')
    #     ax1.set_ylabel('Deposition volume [mm³]')
    #     plt.savefig(os.path.join(plot_dir, run +'_dep.png'), dpi=200)
    #     plt.show()
    # else:
    #     pass


if run_mode==2:
    # Print vulume teporal scale report
    volume_temp_scale_report_header = 'run name, B_dep [min], SD(B_dep) [min], B_sco [min], SD(B_sco) [min]'
    # Write temporl scale report as:
    # run name, B_dep, SD(B_dep), B_sco, SD(B_sco)
    with open(os.path.join(report_dir, 'volume_temp_scale_report.txt'), 'w') as fp:
        fp.write(volume_temp_scale_report_header)
        fp.writelines(['\n'])
        for i in range(0,len(RUNS)):
            for j in range(0, volume_temp_scale_report.shape[1]+1):
                if j == 0:
                    fp.writelines([RUNS[i]+', '])
                else:
                    fp.writelines(["%.3f, " % float(volume_temp_scale_report[i,j-1])])
            fp.writelines(['\n'])
    fp.close()

    # Print morphW teporal scale report
    morphW_temp_scale_report_header = 'run name, B_morphW [min], SD(B_morphW) [min]'
    # Write morphW temporl scale report as:
    # run name, B_morphW, SD(B_morphW)
    with open(os.path.join(report_dir, 'morphW_temp_scale_report.txt'), 'w') as fp:
        fp.write(morphW_temp_scale_report_header)
        fp.writelines(['\n'])
        for i in range(0,len(RUNS)):
            for j in range(0, morphW_temp_scale_report.shape[1]+1):
                if j == 0:
                    fp.writelines([RUNS[i]+', '])
                else:
                    fp.writelines(["%.3f, " % float(morphW_temp_scale_report[i,j-1])])
            fp.writelines(['\n'])
    fp.close()

    if DEM_analysis_mode==1:
        engelund_model_report_header = 'run name, D [m], Q [m^3/s], Wwet/W [-]'
        # Write temporl scale report as:
        # run name, B_dep, SD(B_dep), B_sco, SD(B_sco)
        with open(os.path.join(report_dir, 'engelund_model_report.txt'), 'w') as fp:
            fp.write(engelund_model_report_header)
            fp.writelines(['\n'])
            for i in range(0,len(RUNS)):
                for j in range(0, engelund_model_report.shape[1]+1):
                    if j == 0:
                        fp.writelines([RUNS[i]+', '])
                    elif j==2:
                        fp.writelines(["%.5f, " % float(engelund_model_report[i,j-1])])
                    else:
                        fp.writelines(["%.3f, " % float(engelund_model_report[i,j-1])])
                fp.writelines(['\n'])
        fp.close()



# Create morphWact/W runs boxplot
# Define active width matrix
morphWact_matrix=np.zeros((len(RUNS), int(np.max(morphWact_dim))))
for i in range(0,len(RUNS)):
    data=np.loadtxt(os.path.join(report_dir, RUNS[i] + '_morphWact_array.txt'), delimiter=',')
    morphWact_matrix[i,:len(data)]=data

# Set zero as np.nan
morphWact_matrix = np.where(morphWact_matrix==0, np.nan, morphWact_matrix)

# Multiple boxplot
fig, ax = plt.subplots(dpi=80, figsize=(10,6))
fig.suptitle('Dimensionless morphological active width', fontsize = 18)
for i in range(0, len(RUNS)):
    bplot=ax.boxplot(morphWact_matrix[i,:][~np.isnan(morphWact_matrix[i,:])]/dim_y, positions=[i], widths=0.5) # Data were filtered by np.nan values
ax.yaxis.grid(True)
ax.set_xlabel('Runs', fontsize=12)
ax.set_ylabel('morphWact/W [-]', fontsize=12)
plt.xticks(np.arange(0,len(RUNS), 1), RUNS)
plt.savefig(os.path.join(plot_dir, 'morphWact_boxplot.png'), dpi=200)
plt.show()



end = time.time()
print()
print('Execution time: ', (end-start), 's')
