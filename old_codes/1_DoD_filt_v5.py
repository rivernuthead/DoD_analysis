#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 09:44:30 2021

@author: erri
"""
import os
import shutil
import time
import numpy as np
import cv2
from skimage import morphology
from PIL import Image
import matplotlib.pyplot as plt
import PyPDF2
from PyPDF2 import PdfFileMerger, PdfFileReader, PdfFileWriter
from DoD_analysis_functions_4 import *
from morph_quantities_func_v2 import morph_quantities

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
    0 = runs in the runs list
    1 = one run at time
    2 = bath process 'all the runs in the folder'
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
DoD_plot_save_mode:
    0 = do not save DoD plots
    1 = save plot
DoD_plot_show_mode:
    0 = do not show DoD plots
    1 = show DoD plots
'''
run_mode = 0
mask_mode = 1
process_mode = 1
filt_analysis = 1 # Print the morphological metrics for each filtering process stage

# data_interpolation_mode = 0
# save_plot_mode = 1
# DoD_plot_mode = 0
# DoD_plot_save_mode = 1
# DoD_plot_show_mode = 0

# SINGLE RUN NAME
run = 'q20r9'
# ARRAY OF RUNS
# runs = ['q07_1', 'q10_2', 'q15_2', 'q15_3', 'q20_2']
# runs = ['q07_1']
# runs = ['q10_2', 'q15_2', 'q15_3', 'q20_2']
# runs = ['q10_3', 'q10_4']
runs = ['q10_2']

# Set DEM single name to perform process to specific DEM
if len(run) ==1:
    DEM1_single_name = 'matrix_bed_norm_' + str(run) +'s'+'0'+'.txt' # DEM1 name
    DEM2_single_name = 'matrix_bed_norm_' + str(run) +'s'+'1'+'.txt' # DEM2 name

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
out_dir = os.path.join(home_dir, 'output')
plot_out_dir = os.path.join(home_dir, 'plot')

# Create folders
if not(os.path.exists(out_dir)):
    os.mkdir(out_dir)
if not(os.path.exists(plot_out_dir)):
    os.mkdir(plot_out_dir)
    
DoDs_dir = os.path.join(home_dir, 'DoDs')


run_dir = os.path.join(home_dir, 'surveys')



# Create the run name list
RUNS=[]
if run_mode==0:
    RUNS=runs
elif run_mode==2: # batch run mode
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

# # Define Engelund Gauss model report matrix:
# # D [m], Q [m^3/s], Wwet/W [-]
# engelund_model_report=np.zeros((len(RUNS),3))

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
    report_dir = os.path.join(home_dir, 'output', run)
    plot_dir = os.path.join(home_dir, 'plot', run)
    path_out = os.path.join(home_dir, 'DoDs', 'DoDs_'+run) # path where to save DoDs
    DoDs_plot = os.path.join(home_dir, 'output', 'DoDs_maps', run)
    
    # Save a report with xData as real time in minutes and the value of scour and deposition volumes for each runs
    # Check if the file already exists
    if os.path.exists(os.path.join(report_dir, 'volume_over_time.txt')):
        os.remove(os.path.join(report_dir, 'volume_over_time.txt'))
    else:
        pass

    # CREATE FOLDERS
    if not(os.path.exists(report_dir)):
        os.mkdir(report_dir)
    if not(os.path.exists(DoDs_dir)):
        os.mkdir(DoDs_dir)
    if os.path.exists(os.path.join(DoDs_dir, 'DoDs_'+run)): # If the directory exist, remove it to clean all the old data
        shutil.rmtree(os.path.join(DoDs_dir, 'DoDs_'+run), ignore_errors=False, onerror=None)
    if not(os.path.exists(os.path.join(DoDs_dir, 'DoDs_'+run))): # If the directory does not exist, create it
        os.mkdir(os.path.join(DoDs_dir, 'DoDs_'+run))
    if not(os.path.exists(os.path.join(DoDs_dir, 'DoDs_stack'))):
        os.mkdir(os.path.join(DoDs_dir, 'DoDs_stack'))
    if not(os.path.exists(plot_dir)):
        os.mkdir(plot_dir)
    if not(os.path.exists(os.path.join(home_dir, 'output', 'DoDs_maps'))):
        os.mkdir(os.path.join(home_dir, 'output', 'DoDs_maps'))
    if not(os.path.exists(os.path.join(home_dir, 'output', 'DoDs_maps', run))):
        os.mkdir(os.path.join(home_dir, 'output', 'DoDs_maps', run))



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
    morphWact_values_dep=[] # morphWact values for each section of all the DoD
    morphWact_values_sco=[] # morphWact values for each section of all the DoD
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
    matrix_Wact_sco=np.zeros((len(files)+3, len(files)+3)) # Active width report matrix
    matrix_Wact_dep=np.zeros((len(files)+3, len(files)+3)) # Active width report matrix
    matrix_Wact_IIIquantile=np.zeros((len(files)-1, len(files)+1)) # III quantile active width report matrix
    matrix_Wact_Iquantile=np.zeros((len(files)-1, len(files)+1)) # I quantile active width report matrix
    matrix_Wact_IIIquantile_dep=np.zeros((len(files)-1, len(files)+1)) # III quantile active width report matrix
    matrix_Wact_Iquantile_dep=np.zeros((len(files)-1, len(files)+1)) # I quantile active width report matrix
    matrix_Wact_IIIquantile_sco=np.zeros((len(files)-1, len(files)+1)) # III quantile active width report matrix
    matrix_Wact_Iquantile_sco=np.zeros((len(files)-1, len(files)+1)) # I quantile active width report matrix
    matrix_act_thickness = np.zeros((len(files)-1, len(files)+1)) # Matrix where collect total active thickness data
    matrix_act_thickness_dep = np.zeros((len(files)-1, len(files)+1)) # Matrix where collect deposition active thickness data
    matrix_act_thickness_sco = np.zeros((len(files)-1, len(files)+1)) # Matrix where collect scour active thickness data
    matrix_act_volume = np.zeros((len(files)-1, len(files)+1)) # Matrix where collect volume data

    matrix_DEM_analysis = np.zeros((len(files), len(files)))
    
    # Analysis on the effect of the spatial filter on the morphological changes
    # Initialize array
    DoD_raw_morph_quant = []
    DoD_filt_mean_morph_quant = []
    DoD_filt_isol_morph_quant = []
    DoD_filt_fill_morph_quant = []
    DoD_filt_nature_morph_quant = []
    DoD_filt_isol2_morph_quant = []
    DoD_filt_ult_morph_quant = []

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

#%%
    ###########################################################################
    # LOOP OVER ALL DEMs COMBINATIONS
    ###########################################################################
    nn=0
    # Perform difference between DEMs over all the possible combination of surveys in the survey directory
    for h in range (0, len(files)-1):
        for k in range (0, len(files)-1-h):
            print(h)
            nn+=1
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
            
            print(run)
            print('=========')
            print(DoD_name[:-1])
            print('=========')
            
            # TODO UPDATE
            delta=int(DEM2_num)-int(DEM1_num) # Calculate delta between DEM
            
            print('delta = ', delta)
            print('----------')
            print()
            
            # # Setup output folder
            # output_name = 'script_outputs_' + DEM2_DoD_name + '-' + DEM1_DoD_name # Set outputs name

            # Set DoD outputs directory where to save DoD as ASCII grid and numpy matrix
            
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
            header[0] = header[0].replace(header[0][18:25], ' '+str(int(dim_y*px_y))+str('.00'))
            header[1] = header[1].replace(header[1][17:25], '    '+str('0.00'))
            header[2] = header[2].replace(header[2][17:25], str(int(dim_x*px_x))+str('.00'))
            header[3] = header[3].replace(header[3][17:25], '    '+str('0.00'))
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
            # Scale array for plotting
            DoD_raw_plot = rescaling_plot(DoD_raw)
            
            # Creating GIS readable DoD array (np.nan as -999)
            DoD_raw_gis = np.zeros(DoD_raw.shape)
            DoD_raw_gis = np.where(np.isnan(DoD_raw), NaN, DoD_raw)
            


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
            
            
            # 1- PERFORM DOMAIN-WIDE WEIGHTED AVERAGE:
            # -------------------------------------
            
            # kernel=np.array([[1],
            #                  [1],
            #                  [2],
            #                  [1],
            #                  [1]])
            
            kernel=np.array([[1],
                             [2],
                             [1]])
            ker=kernel/np.sum(kernel)
            
            DoD_filt_mean=cv2.filter2D(src=DoD_raw,ddepth=-1, kernel=ker)
            DoD_filt_mean_gis = np.where(np.isnan(DoD_filt_mean), NaN, DoD_filt_mean)
            DoD_filt_mean_plot = rescaling_plot(DoD_filt_mean)
            
            
            # 2- PERFORM UNDER THRESHOLD ZEROING:
            DoD_filt_thrs = np.where(np.abs(DoD_filt_mean)<=thrs_1, 0, DoD_filt_mean)
            DoD_filt_thrs_gis = np.where(np.isnan(DoD_filt_thrs), NaN, DoD_filt_thrs)
            # Scale array for plotting
            DoD_filt_thrs_plot = rescaling_plot(DoD_filt_thrs)
            
            
            # PLOT A CROSS-SECTION BEFORE AND AFTER THE AVERAGE PROCESS
            plot1 = plt.plot(DoD_raw[:,50], label = 'raw')
            plot2 = plt.plot(DoD_filt_mean[:,50], label = 'avg')
            plot3 = plt.plot(DoD_filt_thrs[:,50], label = 'thrs')
            # Set title and show the plot
            plt.title(run + ' - ' + DoD_name)
            plt.legend()
            # plt.savefig(os.path.join(DoDs_plot, DoD_name + 'plot.png'), dpi=600 )
            plt.savefig(os.path.join(DoDs_plot, DoD_name + 'section_plot.pdf'), dpi=600 )
            plt.show()
            
            
            # 3- PERFORM ISOLATED PIXEL REMOVAL:
            #--------------------------------------------------
            # After trimming al np.nan values, counter represent the number of
            # pixel not equal to zero in the DoD_filt_mean matrix
            counter0 = np.count_nonzero(DoD_filt_thrs[np.logical_not(np.isnan(DoD_filt_thrs))])
            
            # Perform the forst iteration
            DoD_filt_isol = remove_small_objects(DoD_filt_thrs, 8, 1)
            
            # After trimming al np.nan values, counter represent the number of
            # pixel not equal to zero of the DoD_filt_isol matrix
            counter1 = np.count_nonzero(DoD_filt_isol[np.logical_not(np.isnan(DoD_filt_isol))])
            # Perform the isolated_killer procedure until the number of active
            # pixel will not change anymore 
            while counter0-counter1!=0:
                # Filtering...
                DoD_filt_isol = remove_small_objects(DoD_filt_isol, 8, 1)
                # Update counters:
                counter0 = counter1
                counter1 = np.count_nonzero(DoD_filt_isol[np.logical_not(np.isnan(DoD_filt_isol))])
            
            # Convert matrix to be GIS-readable
            DoD_filt_isol_gis = np.where(np.isnan(DoD_filt_isol), NaN, DoD_filt_isol)
            
            # Scale array for plotting
            DoD_filt_isol_plot = rescaling_plot(DoD_filt_isol)

            
            # 4- PERFORM PITTS FILLING PROCEDURE:
            #---------------------------------------
            # Initialize the counter from the previous step of the filtering process
            counter0 = np.count_nonzero(DoD_filt_isol[np.logical_not(np.isnan(DoD_filt_isol))])
            # Perform the first step of the filling procedure
            DoD_filt_fill, matrix_target = fill_small_holes(DoD_filt_isol, avg_target_kernel=5, area_threshold=8, connectivity=1, filt_threshold=thrs_1)
            # Calculate the current counter
            counter1 = np.count_nonzero(DoD_filt_fill[np.logical_not(np.isnan(DoD_filt_fill))])
            # Perform the loop of the filtering process
            while counter0-counter1!=0:
                # Filtering...
                DoD_filt_fill, matrix_target = fill_small_holes(DoD_filt_fill, avg_target_kernel=5, area_threshold=8, connectivity=1, filt_threshold=thrs_1)
                # Update counters:
                counter0=counter1
                counter1 = np.count_nonzero(DoD_filt_fill[np.logical_not(np.isnan(DoD_filt_fill))])   
            
            # Convert matrix to be GIS-readable
            DoD_filt_fill_gis = np.where(np.isnan(DoD_filt_fill), NaN, DoD_filt_fill)
                
            # Scale array for plotting
            DoD_filt_fill_plot = rescaling_plot(DoD_filt_fill)
            
            
            # 5- PERFORM NATURE CHECKER PIXEL PROCEDURE:
            #----------------------------------------
            # TODO THIS IS NOT ACTIVE!!
            DoD_filt_nature, DoD_filt_nature_gis = nature_checker(DoD_filt_fill, thrs_nature, 1, NaN)
            DoD_filt_nature_plot = rescaling_plot(DoD_filt_nature)
            
            
            # 6- RE-PERFORM ISOLATED PIXEL REMOVAL:
            #-----------------------------------------------------
            # After trimming al np.nan values, counter represent the number of
            # pixel not equal to zero o fthe DoD_filt_fill matrix
            counter0 = np.count_nonzero(DoD_filt_nature[np.logical_not(np.isnan(DoD_filt_nature))])
            
            # Perform the very first isolated_killer procedure
            DoD_filt_isol2 = remove_small_objects(DoD_filt_nature, 15, 1)
            
            # After trimming al np.nan values, counter represent the number of
            # pixel not equal to zero of the DoD_filt_isol2 matrix
            counter1 = np.count_nonzero(DoD_filt_isol2[np.logical_not(np.isnan(DoD_filt_isol2))])
            # Perform the isolated_killer procedure until the number of active
            # pixel will not change anymore

            while counter0-counter1!=0:
                # Filtering...
                DoD_filt_isol2 = remove_small_objects(DoD_filt_isol2, 15, 1)
                # Update counters:
                counter0 = counter1
                counter1 = np.count_nonzero(DoD_filt_isol2[np.logical_not(np.isnan(DoD_filt_isol2))])

            DoD_filt_isol2_gis = np.where(np.isnan(DoD_filt_isol2), NaN, DoD_filt_isol2)
            
            DoD_filt_isol2_plot = rescaling_plot(DoD_filt_isol2)
            
            
            
            # 6- PERFORM ISOLATED PIXEL REMOVAL:
            #-----------------------------------------------------
            DoD_filt_isol3      = test(DoD_filt_isol2)
            DoD_filt_isol3_gis  = np.where(np.isnan(DoD_filt_isol3), NaN, DoD_filt_isol3)
            DoD_filt_isol3_plot = rescaling_plot(DoD_filt_isol3)
            
            
            
            # 7- RE-PERFORM ISOLATED ANISOTROPIC PIXEL REMOVAL:
            #-----------------------------------------------------
            # After trimming al np.nan values, counter represent the number of
            # pixel not equal to zero o fthe DoD_filt_fill matrix
            counter0 = np.count_nonzero(DoD_filt_isol3[np.logical_not(np.isnan(DoD_filt_isol3))])
            
            # Perform the first iteration
            DoD_filt_isol4 = test2(DoD_filt_isol3)
            
            # After trimming al np.nan values, counter represent the number of
            # pixel not equal to zero of the DoD_filt_isol2 matrix
            counter1 = np.count_nonzero(DoD_filt_isol4[np.logical_not(np.isnan(DoD_filt_isol4))])
            # Perform the isolated_killer procedure until the number of active
            # pixel will not change anymore
            
            while counter0-counter1!=0:
                # Filtering...
                DoD_filt_isol4 = test2(DoD_filt_isol3)
                # Update counters:
                counter0 = counter1
                counter1 = np.count_nonzero(DoD_filt_isol4[np.logical_not(np.isnan(DoD_filt_isol4))])
                
            DoD_filt_ult = DoD_filt_isol4
            
            # Set all the value between +/- 2mm at 2mm and keep zero as zero
            DoD_filt_ult = np.where(np.logical_and(abs(DoD_filt_isol4)<thrs_1, DoD_filt_isol4!=0), thrs_1,DoD_filt_isol4)
            
            # DoD_filt_ult       = test2(DoD_filt_isol4)
            # DoD_filt_ult = np.where(DoD_filt_ult>0, remove_small_objects(DoD_filt_ult>0, 40, 1), DoD_filt_ult)
            # DoD_filt_ult = np.where(DoD_filt_ult<0, remove_small_objects(DoD_filt_ult<0, 40, 1), DoD_filt_ult)
            # DoD_filt_ult = DoD_filt_ult*DoD_filt_isol4
            DoD_filt_ult_gis   = np.where(np.isnan(DoD_filt_ult), NaN, DoD_filt_ult)
            
            
            # Scale array real size
            DoD_filt_ult_plot = rescaling_plot(DoD_filt_ult)
            
            
            ################
            # Filtering process visual check
            ################
            
            img2 = plt.imshow(DoD_raw_plot, cmap='binary', alpha=1.0, vmin=-20, vmax=+20, interpolation_stage='rgba')
            img1 = plt.imshow(DoD_filt_mean_plot, cmap='RdBu', alpha=1.0, vmin=-20, vmax=+20, interpolation_stage='rgba')
            # Set title and show the plot
            plt.title(run + ' - ' + DoD_name[:-1])
            plt.axis('off')
            # plt.savefig(os.path.join(DoDs_plot, DoD_name + 'raw-thrs.png'), dpi=2000)
            plt.savefig(os.path.join(DoDs_plot, DoD_name + 'raw-mean.pdf'), dpi=2000)
            plt.show()
            
            img2 = plt.imshow(DoD_filt_mean_plot, cmap='binary', alpha=1.0, vmin=-20, vmax=+20, interpolation_stage='rgba')
            img1 = plt.imshow(DoD_filt_thrs_plot, cmap='RdBu', alpha=1.0, vmin=-20, vmax=+20, interpolation_stage='rgba')
            # Set title and show the plot
            plt.title(run + ' - ' + DoD_name[:-1])
            plt.axis('off')
            # plt.savefig(os.path.join(DoDs_plot, DoD_name + 'raw-thrs.png'), dpi=2000)
            plt.savefig(os.path.join(DoDs_plot, DoD_name + 'mean_thrs.pdf'), dpi=2000)
            plt.show()
            
            img2 = plt.imshow(DoD_filt_thrs_plot, cmap='binary', alpha=1.0, vmin=-20, vmax=+20, interpolation_stage='rgba')
            img1 = plt.imshow(DoD_filt_isol_plot, cmap='RdBu', alpha=1.0, vmin=-20, vmax=+20, interpolation_stage='rgba')
            # Set title and show the plot
            plt.title(run + ' - ' + DoD_name[:-1])
            plt.axis('off')
            # plt.savefig(os.path.join(DoDs_plot, DoD_name + 'thrs-isol.png'), dpi=2000)
            plt.savefig(os.path.join(DoDs_plot, DoD_name + 'thrs-isol.pdf'), dpi=2000)
            plt.show()
            
            img2 = plt.imshow(DoD_filt_isol_plot, cmap='binary', alpha=1.0, vmin=-20, vmax=+20, interpolation_stage='rgba')
            img1 = plt.imshow(DoD_filt_fill_plot, cmap='RdBu', alpha=0.5, vmin=-20, vmax=+20, interpolation_stage='rgba')
            # Set title and show the plot
            plt.title(run + ' - ' + DoD_name[:-1])
            plt.axis('off')
            # plt.savefig(os.path.join(DoDs_plot, DoD_name + 'isol-fill.png'), dpi=2000)
            plt.savefig(os.path.join(DoDs_plot, DoD_name + 'isol-fill.pdf'), dpi=2000)
            plt.show()
            
            img2 = plt.imshow(DoD_filt_fill_plot, cmap='binary', alpha=1.0, vmin=-20, vmax=+20, interpolation_stage='rgba')
            img1 = plt.imshow(DoD_filt_nature_plot, cmap='RdBu', alpha=1.0, vmin=-20, vmax=+20, interpolation_stage='rgba')
            # Set title and show the plot
            plt.title(run + ' - ' + DoD_name[:-1])
            plt.axis('off')
            # plt.savefig(os.path.join(DoDs_plot, DoD_name + 'fill-nature.png'), dpi=2000)
            plt.savefig(os.path.join(DoDs_plot, DoD_name + 'fill-nature.pdf'), dpi=2000)
            plt.show()
 
            img2 = plt.imshow(DoD_filt_nature_plot, cmap='binary', alpha=1.0, vmin=-20, vmax=+20, interpolation_stage='rgba')
            img1 = plt.imshow(DoD_filt_isol2_plot, cmap='RdBu', alpha=1.0, vmin=-20, vmax=+20, interpolation_stage='rgba')
            # Set title and show the plot
            plt.title(run + ' - ' + DoD_name[:-1])
            plt.axis('off')
            # plt.savefig(os.path.join(DoDs_plot, DoD_name + 'nature-isol2.png'), dpi=2000)
            plt.savefig(os.path.join(DoDs_plot, DoD_name + 'nature-isol2.pdf'), dpi=2000)
            plt.show()
            
            img2 = plt.imshow(DoD_filt_isol2_plot, cmap='binary', alpha=1.0, vmin=-20, vmax=+20, interpolation_stage='rgba')
            img1 = plt.imshow(DoD_filt_isol3_plot, cmap='RdBu', alpha=1.0, vmin=-20, vmax=+20, interpolation_stage='rgba')
            # Set title and show the plot
            plt.title(run + ' - ' + DoD_name[:-1])
            plt.axis('off')
            # plt.savefig(os.path.join(DoDs_plot, DoD_name + 'isol2-isol3.png'), dpi=2000)
            plt.savefig(os.path.join(DoDs_plot, DoD_name + 'isol2-isol3.pdf'), dpi=2000)
            plt.show()
            
            img2 = plt.imshow(DoD_filt_isol3_plot, cmap='binary', alpha=1.0, vmin=-20, vmax=+20, interpolation_stage='rgba')
            img1 = plt.imshow(DoD_filt_ult_plot, cmap='RdBu', alpha=1.0, vmin=-20, vmax=+20, interpolation_stage='rgba')
            # Set title and show the plot
            plt.title(run + ' - ' + DoD_name[:-1])
            plt.axis('off')
            # plt.savefig(os.path.join(DoDs_plot, DoD_name + 'isol3-isol4.png'), dpi=2000)
            plt.savefig(os.path.join(DoDs_plot, DoD_name + 'isol3-ult.pdf'), dpi=2000)
            plt.show()
            
            
            

            
            # Plot the result for double check
            img2 = plt.imshow(DoD_filt_thrs_plot, cmap='binary', alpha=0.4, vmin=-20, vmax=+20)
            # img1 = plt.imshow(np.where(DoD_filt_ult_plot==0, np.nan, DoD_filt_ult_plot), cmap='RdBu', alpha=1.0, vmin=-20, vmax=+20)
            img1 = plt.imshow(DoD_filt_ult_plot, cmap='RdBu', alpha=1.0, vmin=-20, vmax=+20, interpolation_stage='rgba')

            # Set title and show the plot
            plt.title(run + ' - ' + DoD_name[:-1])
            plt.axis('off')
            plt.savefig(os.path.join(DoDs_plot, DoD_name + '_plot.png'), dpi=2000)
            plt.savefig(os.path.join(DoDs_plot, DoD_name + '_plot.pdf'), dpi=2000)
            
            
            # CREATE A PDF REPORT TO COLLECT ALL PLOTS
            if nn==1:
                plt.savefig(os.path.join(DoDs_plot,run +'_merged_report_plot.pdf'), dpi=1200 )
                
            plt.show()
            

            if nn>1:
                merger = PyPDF2.PdfMerger()

                # Open and append the existing PDF
                with open(os.path.join(DoDs_plot,run +'_merged_report_plot.pdf'), "rb") as existing_file:
                    merger.append(existing_file)

                # Open and append the new PDF chart
                with open(os.path.join(DoDs_plot, DoD_name + '_plot.pdf'), "rb") as chart_file:
                    merger.append(chart_file)

                # Save the merged PDF
                with open(os.path.join(DoDs_plot,run +'_merged_report_plot.pdf'), "wb") as merged_file:
                    merger.write(merged_file)
            

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

#%%         ###################################################################
            # MORPHOLOGICAL QUANTITIES:
            ###################################################################
            
            tot_vol, sum_vol, dep_vol, sco_vol, morph_act_area, morph_act_area_dep, morph_act_area_sco, act_width_mean, act_width_mean_dep, act_width_mean_sco, act_thickness, act_thickness_dep, act_thickness_sco, bri = morph_quantities(DoD_filt_ult)
            
            ###################################################################
            # Filtering process stage analysis
            if filt_analysis == 1:
                # Analysis to investigate the role of the application of spatial filters at the DoD
                # in the morphological changes calculation
                if delta==1:
                    # DoD_raw
                    tot_vol, sum_vol, dep_vol, sco_vol, morph_act_area, morph_act_area_dep, morph_act_area_sco, act_width_mean, act_width_mean_dep, act_width_mean_sco, act_thickness, act_thickness_dep, act_thickness_sco, bri = morph_quantities(DoD_raw)
                    if len(DoD_raw_morph_quant)==0:
                        DoD_raw_morph_quant=np.append(DoD_raw_morph_quant, (tot_vol, sum_vol, dep_vol, sco_vol, morph_act_area, morph_act_area_dep, morph_act_area_sco, act_width_mean, act_width_mean_dep, act_width_mean_sco, act_thickness, act_thickness_dep, act_thickness_sco, bri))
                    else:
                        DoD_raw_morph_quant=np.vstack((DoD_raw_morph_quant, np.hstack((tot_vol, sum_vol, dep_vol, sco_vol, morph_act_area, morph_act_area_dep, morph_act_area_sco, act_width_mean, act_width_mean_dep, act_width_mean_sco, act_thickness, act_thickness_dep, act_thickness_sco, bri))))
                    
                    # DoD_filt_mean
                    tot_vol, sum_vol, dep_vol, sco_vol, morph_act_area, morph_act_area_dep, morph_act_area_sco, act_width_mean, act_width_mean_dep, act_width_mean_sco, act_thickness, act_thickness_dep, act_thickness_sco, bri = morph_quantities(DoD_filt_mean)
                    if len(DoD_filt_mean_morph_quant)==0:
                        DoD_filt_mean_morph_quant=np.append(DoD_filt_mean_morph_quant, (tot_vol, sum_vol, dep_vol, sco_vol, morph_act_area, morph_act_area_dep, morph_act_area_sco, act_width_mean, act_width_mean_dep, act_width_mean_sco, act_thickness, act_thickness_dep, act_thickness_sco, bri))
                    else:
                        DoD_filt_mean_morph_quant=np.vstack((DoD_filt_mean_morph_quant, np.hstack((tot_vol, sum_vol, dep_vol, sco_vol, morph_act_area, morph_act_area_dep, morph_act_area_sco, act_width_mean, act_width_mean_dep, act_width_mean_sco, act_thickness, act_thickness_dep, act_thickness_sco, bri))))
                    
                    # DoD_filt_isol
                    tot_vol, sum_vol, dep_vol, sco_vol, morph_act_area, morph_act_area_dep, morph_act_area_sco, act_width_mean, act_width_mean_dep, act_width_mean_sco, act_thickness, act_thickness_dep, act_thickness_sco, bri = morph_quantities(DoD_filt_isol)
                    if len(DoD_filt_isol_morph_quant)==0:
                        DoD_filt_isol_morph_quant=np.append(DoD_filt_isol_morph_quant, (tot_vol, sum_vol, dep_vol, sco_vol, morph_act_area, morph_act_area_dep, morph_act_area_sco, act_width_mean, act_width_mean_dep, act_width_mean_sco, act_thickness, act_thickness_dep, act_thickness_sco, bri))
                    else:
                        DoD_filt_isol_morph_quant=np.vstack((DoD_filt_isol_morph_quant, np.hstack((tot_vol, sum_vol, dep_vol, sco_vol, morph_act_area, morph_act_area_dep, morph_act_area_sco, act_width_mean, act_width_mean_dep, act_width_mean_sco, act_thickness, act_thickness_dep, act_thickness_sco, bri))))
                    
                    # DoD_filt_fill
                    tot_vol, sum_vol, dep_vol, sco_vol, morph_act_area, morph_act_area_dep, morph_act_area_sco, act_width_mean, act_width_mean_dep, act_width_mean_sco, act_thickness, act_thickness_dep, act_thickness_sco, bri = morph_quantities(DoD_filt_fill)
                    if len(DoD_filt_fill_morph_quant)==0:
                        DoD_filt_fill_morph_quant=np.append(DoD_filt_fill_morph_quant, (tot_vol, sum_vol, dep_vol, sco_vol, morph_act_area, morph_act_area_dep, morph_act_area_sco, act_width_mean, act_width_mean_dep, act_width_mean_sco, act_thickness, act_thickness_dep, act_thickness_sco, bri))
                    else:
                        DoD_filt_fill_morph_quant=np.vstack((DoD_filt_fill_morph_quant, np.hstack((tot_vol, sum_vol, dep_vol, sco_vol, morph_act_area, morph_act_area_dep, morph_act_area_sco, act_width_mean, act_width_mean_dep, act_width_mean_sco, act_thickness, act_thickness_dep, act_thickness_sco, bri))))
                    
                    # DoD_filt_nature
                    tot_vol, sum_vol, dep_vol, sco_vol, morph_act_area, morph_act_area_dep, morph_act_area_sco, act_width_mean, act_width_mean_dep, act_width_mean_sco, act_thickness, act_thickness_dep, act_thickness_sco, bri = morph_quantities(DoD_filt_nature)
                    if len(DoD_filt_nature_morph_quant)==0:
                        DoD_filt_nature_morph_quant=np.append(DoD_filt_nature_morph_quant, (tot_vol, sum_vol, dep_vol, sco_vol, morph_act_area, morph_act_area_dep, morph_act_area_sco, act_width_mean, act_width_mean_dep, act_width_mean_sco, act_thickness, act_thickness_dep, act_thickness_sco, bri))
                    else:
                        DoD_filt_nature_morph_quant=np.vstack((DoD_filt_nature_morph_quant, np.hstack((tot_vol, sum_vol, dep_vol, sco_vol, morph_act_area, morph_act_area_dep, morph_act_area_sco, act_width_mean, act_width_mean_dep, act_width_mean_sco, act_thickness, act_thickness_dep, act_thickness_sco, bri))))
                    
                    # DoD_filt_isol2
                    tot_vol, sum_vol, dep_vol, sco_vol, morph_act_area, morph_act_area_dep, morph_act_area_sco, act_width_mean, act_width_mean_dep, act_width_mean_sco, act_thickness, act_thickness_dep, act_thickness_sco, bri = morph_quantities(DoD_filt_isol2)
                    if len(DoD_filt_isol2_morph_quant)==0:
                        DoD_filt_isol2_morph_quant=np.append(DoD_filt_isol2_morph_quant, (tot_vol, sum_vol, dep_vol, sco_vol, morph_act_area, morph_act_area_dep, morph_act_area_sco, act_width_mean, act_width_mean_dep, act_width_mean_sco, act_thickness, act_thickness_dep, act_thickness_sco, bri))
                    else:
                        DoD_filt_isol2_morph_quant=np.vstack((DoD_filt_isol2_morph_quant, np.hstack((tot_vol, sum_vol, dep_vol, sco_vol, morph_act_area, morph_act_area_dep, morph_act_area_sco, act_width_mean, act_width_mean_dep, act_width_mean_sco, act_thickness, act_thickness_dep, act_thickness_sco, bri))))
                    
                    # DoD_filt_ult
                    tot_vol, sum_vol, dep_vol, sco_vol, morph_act_area, morph_act_area_dep, morph_act_area_sco, act_width_mean, act_width_mean_dep, act_width_mean_sco, act_thickness, act_thickness_dep, act_thickness_sco, bri = morph_quantities(DoD_filt_ult)
                    if len(DoD_filt_ult_morph_quant)==0:
                        DoD_filt_ult_morph_quant=np.append(DoD_filt_ult_morph_quant, (tot_vol, sum_vol, dep_vol, sco_vol, morph_act_area, morph_act_area_dep, morph_act_area_sco, act_width_mean, act_width_mean_dep, act_width_mean_sco, act_thickness, act_thickness_dep, act_thickness_sco, bri))
                    else:
                        DoD_filt_ult_morph_quant=np.vstack((DoD_filt_ult_morph_quant, np.hstack((tot_vol, sum_vol, dep_vol, sco_vol, morph_act_area, morph_act_area_dep, morph_act_area_sco, act_width_mean, act_width_mean_dep, act_width_mean_sco, act_thickness, act_thickness_dep, act_thickness_sco, bri))))
                

            # Fill arrays with the values of the total, sum, deposition and scour volume
            # Append values to output data array
            volumes_array = np.append(volumes_array, tot_vol)
            sum_array = np.append(sum_array, sum_vol)
            dep_array = np.append(dep_array, dep_vol)
            sco_array = np.append(sco_array, sco_vol)
        
            ###################################################################
            # Active_pixel analysis
            ###################################################################
            
            # MORPHOLOGICAL ACTIVE AREA MAP
            # Create the map of of activity 
            morph_act_area_array = np.append(morph_act_area_array, morph_act_area) # For each DoD, append total active area data
            morph_act_area_array_dep = np.append(morph_act_area_array_dep, morph_act_area_dep) # For each DoD, append deposition active area data
            morph_act_area_array_sco = np.append(morph_act_area_array_sco, morph_act_area_sco) # For each DoD, append scour active area data
            
            # MORPHOLOGICAL ACTIVE WIDTH PROFILE
            # Calculate the morphological active width section by section 
            # Morphological active width streamwise array [number of activ cells]
            act_width_array = np.array([np.nansum(act_px_matrix, axis=0)]) # Array of the crosswise morphological total active width in number of active cells [-]
            act_width_array_dep = np.array([np.nansum(act_px_matrix_dep, axis=0)]) # Array of the crosswise morphological deposition active width in number of active cells [-]
            act_width_array_sco = np.array([np.nansum(act_px_matrix_sco, axis=0)]) # Array of the crosswise morphological scour active width in number of active cells [-]
            
            
            # Morphological active width [mean value, number of activ cells / channel width]
            act_width_mean_array = np.append(act_width_mean_array, act_width_mean/(W/(px_y/1000))) # For each DoD append total active width values [actW/W]
            act_width_mean_array_dep = np.append(act_width_mean_array_dep, act_width_mean_dep/(W/(px_y/1000))) # For each DoD append deposition active width values [actW/W]
            act_width_mean_array_sco = np.append(act_width_mean_array_sco, act_width_mean_sco/(W/(px_y/1000))) # For each DoD append scour active width values [actW/W]
            
            
            #Print results:
            print('Total volume:', "{:.1f}".format(tot_vol))
            print('Sum of deposition and scour volume:', "{:.1f}".format(sum_vol))
            print('Deposition volume:', "{:.1f}".format(dep_vol))
            print('Scour volume:', "{:.1f}".format(sco_vol))
            print('Active thickness [mm]:', act_thickness)
            print('Active thickness of deposition [mm]', act_thickness_dep)
            print('Active thickness of scour [mm]', act_thickness_sco)
            print('Morphological active area (number of active cells): ', "{:.1f}".format(morph_act_area), '[-]')
            print('Morphological deposition active area (number of active cells): ', "{:.1f}".format(morph_act_area_dep), '[-]')
            print('Morphological scour active area (number of active cells): ', "{:.1f}".format(morph_act_area_sco), '[-]')
            print('Morphological active width (mean):', "{:.3f}".format(act_width_mean/(W/(px_y/1000))), 'actW/W [-]')
            print('Morphological deposition active width (mean):', "{:.3f}".format(act_width_mean_dep/(W/(px_y/1000))), 'actW/W [-]')
            print('Morphological scour active width (mean):', "{:.3f}".format(act_width_mean_sco/(W/(px_y/1000))), 'actW/W [-]')
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

            
            # Build up morphWact/W array for the current run boxplot
            # This array contain all the morphWact/W values for all the run repetition in the same line
            # This array contain only adjacent DEMs DoD
            # TODO
            if delta==1:
                morphWact_values = np.append(morphWact_values, act_width_array/(W/(px_y/1000)))
                morphWact_values_dep = np.append(morphWact_values_dep, act_width_array_dep/(W/(px_y/1000)))
                morphWact_values_sco = np.append(morphWact_values_sco, act_width_array_sco/(W/(px_y/1000)))

            # Fill Scour, Deposition and morphWact/W matrix:
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
                matrix_Wact[delta-1,h]=act_width_mean/(W/(px_y/1000))
                matrix_Wact_dep[delta-1,h]=act_width_mean_dep/(W/(px_y/1000))
                matrix_Wact_sco[delta-1,h]=act_width_mean_sco/(W/(px_y/1000))
                
                # Fill last two columns with AVERAGE and STDEV of the corresponding row
                # AVERAGE
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
                matrix_Wact_dep[delta-1,-2]=np.average(matrix_Wact_dep[delta-1,:len(files)-delta])
                matrix_Wact_sco[delta-1,-2]=np.average(matrix_Wact_sco[delta-1,:len(files)-delta])
                
                # Fill last two columns with STDEV of the corresponding row
                # STDEV
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
                matrix_Wact_dep[delta-1,-1]=np.std(matrix_Wact_dep[delta-1,:len(files)-delta])
                matrix_Wact_sco[delta-1,-1]=np.std(matrix_Wact_sco[delta-1,:len(files)-delta])
                
                
                # ACTIVE WIDTH QUANTILE DATA
                # NB: the quantile data has been performed from the streamwise values array of the active width of each cross section
                # Fill III quantile Wact/W matrix:
                matrix_Wact_IIIquantile[delta-1,h]=np.quantile(act_width_array, .75)
                matrix_Wact_IIIquantile[delta-1,-2]=np.min(matrix_Wact_IIIquantile[delta-1,:len(files)-delta])
                matrix_Wact_IIIquantile[delta-1,-1]=np.max(matrix_Wact_IIIquantile[delta-1,:len(files)-delta])
                
                matrix_Wact_IIIquantile_dep[delta-1,h]=np.quantile(act_width_array_dep, .75)
                matrix_Wact_IIIquantile_dep[delta-1,-2]=np.min(matrix_Wact_IIIquantile_dep[delta-1,:len(files)-delta])
                matrix_Wact_IIIquantile_dep[delta-1,-1]=np.max(matrix_Wact_IIIquantile_dep[delta-1,:len(files)-delta])
                
                matrix_Wact_IIIquantile_sco[delta-1,h]=np.quantile(act_width_array_sco, .75)
                matrix_Wact_IIIquantile_sco[delta-1,-2]=np.min(matrix_Wact_IIIquantile_sco[delta-1,:len(files)-delta])
                matrix_Wact_IIIquantile_sco[delta-1,-1]=np.max(matrix_Wact_IIIquantile_sco[delta-1,:len(files)-delta])


                # Fill I quantile Wact/W matrix:
                matrix_Wact_Iquantile[delta-1,h]=np.quantile(act_width_array, .25)
                matrix_Wact_Iquantile[delta-1,-2]=np.min(matrix_Wact_Iquantile[delta-1,:len(files)-delta])
                matrix_Wact_Iquantile[delta-1,-1]=np.max(matrix_Wact_Iquantile[delta-1,:len(files)-delta])
                
                matrix_Wact_Iquantile_dep[delta-1,h]=np.quantile(act_width_array_dep, .25)
                matrix_Wact_Iquantile_dep[delta-1,-2]=np.min(matrix_Wact_Iquantile_dep[delta-1,:len(files)-delta])
                matrix_Wact_Iquantile_dep[delta-1,-1]=np.max(matrix_Wact_Iquantile_dep[delta-1,:len(files)-delta]) 
                
                matrix_Wact_Iquantile_sco[delta-1,h]=np.quantile(act_width_array_sco, .25)
                matrix_Wact_Iquantile_sco[delta-1,-2]=np.min(matrix_Wact_Iquantile_sco[delta-1,:len(files)-delta])
                matrix_Wact_Iquantile_sco[delta-1,-1]=np.max(matrix_Wact_Iquantile_sco[delta-1,:len(files)-delta]) 
                

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

            
            # TODO
            ###################################################################
            # STACK CONSECUTIVE DoDS IN A 3D ARRAY
            ###################################################################
            # Initialize 3D array to stack DoDs
            if h==0 and k==0: # initialize the first array with the DEM shape
                # stack["DoD_stack_{0}".format(run, delta)] = np.zeros([len(files)-1, dim_y, dim_x])
                DoD_stack = np.ones([len(files)-1, dim_y, dim_x, len(files)-1])*NaN
              # DoD_stack[time, X, Y, delta]
            else:
                pass
            
            
            # STACK ALL THE DoDS INSIDE THE 3D ARRAY
            DoD_stack[h,:,:, delta-1] = DoD_filt_ult[:,:]
            DoD_stack = np.where(DoD_stack==NaN, np.nan, DoD_stack)
            
            
            
            # CREATE STACK BOOL
            DoD_stack_bool = np.where(DoD_stack>0, 1, DoD_stack)
            DoD_stack_bool = np.where(DoD_stack_bool<0, -1, DoD_stack_bool)
            
            '''
            DoD1-0  DoD2-0  DoD3-0  DoD4-0  DoD5-0  DoD6-0  DoD7-0  DoD8-0  DoD9-0
            DoD2-1  DoD3-1  DoD4-1  DoD5-1  DoD6-1  DoD7-1  DoD8-1  DoD9-1
            DoD3-2  DoD4-2  DoD5-2  DoD6-2  DoD7-2  DoD8-2  DoD9-2
            DoD4-3  DoD5-3  DoD6-3  DoD7-3  DoD8-3  DoD9-3
            DoD5-4  DoD6-4  DoD7-4  DoD8-4  DoD9-4
            DoD6-5  DoD7-5  DoD8-5  DoD9-5
            DoD7-6  DoD8-6  DoD9-6
            DoD8-7  DoD9-7
            DoD9-8
            '''
            

            ###################################################################
            # DoDs SAVING...
            ###################################################################

            
            # SAVE DOD FILES...
            
            # RAW DoD
            # Print raw DoD in txt file (NaN as np.nan)
            np.savetxt(os.path.join(path_out, DoD_name + 'raw.txt'), DoD_raw, fmt='%0.1f', delimiter='\t')
            # Printing raw DoD in txt file (NaN as -999)
            np.savetxt(os.path.join(path_out, DoD_name + 'raw_gis.txt'), DoD_raw_gis, fmt='%0.1f', delimiter='\t')

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
            # with open(os.path.join(path_out, DoD_name + 'filt_nature_gis.txt')) as f_DoD_nature:
            #     w_DoD_nature = f_DoD_nature.read()
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
                # DoD_nature_gis = w_header + w_DoD_nature
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
            # with open(os.path.join(path_out, DoD_name + 'filt_nature_gis.txt'), 'w') as fp:
            #     fp.write(DoD_nature_gis)
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
    '''
    INPUTS:
        DoD_stack1 : 3D numpy array stack
            Stack on which each 1-step DoD has been saved (with extra domain cell as np.nan)
    OUTPUTS SAVED FILES:
        DoD_stack1 : 3D numpy array stack
            Stack on which DoDs are stored as they are, with np.nan
        DoD_stack1_bool : 3D numpy array stack
            Stack on which DoDs are stored as -1, 0, +1 data, also with np.nan
    '''

    # Save 3D array as binary file
    np.save(os.path.join(DoDs_dir, 'DoDs_stack',"DoD_stack_"+run+".npy"), DoD_stack)
    
    
    # Save 3D "boolean" array as binary file
    np.save(os.path.join(DoDs_dir, 'DoDs_stack',"DoD_stack_bool_"+run+".npy"), DoD_stack_bool)


    # Fill DoD lenght array
    DoD_length_array = np.append(DoD_length_array, DoD_length)



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
    
    # Create Wact scour report matrix
    matrix_Wact_sco=matrix_Wact_sco[:len(files)-1,:] # Fill matrix_Wact with morphological  active width values
    matrix_Wact_sco[:,len(files)-1]=matrix_Wact_Iquantile_sco[:,len(files)-1] # Fill matrix_Wact report with minimum values
    matrix_Wact_sco[:,len(files)]=matrix_Wact_IIIquantile_sco[:,len(files)] # Fill matrix_Wact report with maximum values
    report_Wact_name_sco = os.path.join(report_dir, run +'_morphWact_sco_report.txt')
    np.savetxt(report_Wact_name_sco, matrix_Wact_sco, fmt='%.3f', delimiter=',', newline='\n')
    
    # Create Wact fill report matrix
    matrix_Wact_dep=matrix_Wact_dep[:len(files)-1,:] # Fill matrix_Wact with morphological  active width values
    matrix_Wact_dep[:,len(files)-1]=matrix_Wact_Iquantile_dep[:,len(files)-1] # Fill matrix_Wact report with minimum values
    matrix_Wact_dep[:,len(files)]=matrix_Wact_IIIquantile_dep[:,len(files)] # Fill matrix_Wact report with maximum values
    report_Wact_name_dep = os.path.join(report_dir, run +'_morphWact_dep_report.txt')
    np.savetxt(report_Wact_name_dep, matrix_Wact_dep, fmt='%.3f', delimiter=',', newline='\n')

    # For each runs collect the dimension of the morphWact_array:
    if delta==1:
        morphWact_dim = np.append(morphWact_dim, len(morphWact_values))


    # Create morphWact/W matrix as following:
    # all morphWact/W values are appended in the same line for each line in the morphWact_values array
    # Now a matrix in which all row are all morphWact/W values for each runs is built
    # morphWact_matrix_header = 'run name, morphWact/W [-]'
    # run name, morphWact/W [-]
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
    
    with open(os.path.join(report_dir, run + '_morphWact_array_dep.txt'), 'w') as fp:
        # fp.write(morphWact_matrix_header)
        # fp.writelines(['\n'])
        for i in range(0, len(morphWact_values)):
            if i == len(morphWact_values)-1:
                fp.writelines(["%.3f" % float(morphWact_values_dep[i])])
            else:
                fp.writelines(["%.3f," % float(morphWact_values_dep[i])])
        fp.writelines(['\n'])
    fp.close()
    
    with open(os.path.join(report_dir, run + '_morphWact_array_sco.txt'), 'w') as fp:
        # fp.write(morphWact_matrix_header)
        # fp.writelines(['\n'])
        for i in range(0, len(morphWact_values)):
            if i == len(morphWact_values)-1:
                fp.writelines(["%.3f" % float(morphWact_values_sco[i])])
            else:
                fp.writelines(["%.3f," % float(morphWact_values_sco[i])])
        fp.writelines(['\n'])
    fp.close()



    # # Print a report with xData as real time in minutes and  the value of scour and deposition volumes for each runs
    # Create report matrix as:
    # run
    # time
    # V_dep
    # V_sco
    
    xData1=np.arange(1, len(files), 1)*dt_xnr # Time in Txnr
    yData_sco=np.absolute(matrix_sco[:len(files)-1,0])
    yError_sco=matrix_sco[:len(files)-1,-1]
    yData_dep=np.absolute(matrix_dep[:len(files)-1,0])
    yError_dep=matrix_dep[:len(files)-1,-1]
    yData_act_thickness=matrix_act_thickness[:len(files)-1,0]
    yError_act_thickness=matrix_act_thickness[:len(files)-1,-1]
    
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

    if filt_analysis==1:
        n=0
        for matrix in (DoD_raw_morph_quant, DoD_filt_mean_morph_quant, DoD_filt_isol_morph_quant,DoD_filt_fill_morph_quant,DoD_filt_nature_morph_quant,DoD_filt_isol2_morph_quant,DoD_filt_ult_morph_quant):
            n+=1
            matrix_mean = np.nanmean(matrix, axis=0)
            matrix_std = np.nanstd(matrix, axis=0)
            if n==1:
                matrix_stack = np.vstack((matrix_mean, matrix_std))
            else:
                matrix_stack = np.vstack((matrix_stack,matrix_mean, matrix_std))
            np.savetxt(os.path.join(report_dir, 'morph_quant_report.txt'), matrix_stack, header='tot_vol, sum_vol, dep_vol, sco_vol, morph_act_area, morph_act_area_dep, morph_act_area_sco, act_width_mean, act_width_mean_dep, act_width_mean_sco, act_thickness, act_thickness_dep, act_thickness_sco, bri')
            
        # Save report for the analysis of the effects of spatial filter application on DoD, at different stages.
        np.savetxt(os.path.join(report_dir, 'DoD_raw_morph_quant.txt'), DoD_raw_morph_quant, header='tot_vol, sum_vol, dep_vol, sco_vol, morph_act_area, morph_act_area_dep, morph_act_area_sco, act_width_mean, act_width_mean_dep, act_width_mean_sco, act_thickness, act_thickness_dep, act_thickness_sco, bri')
        np.savetxt(os.path.join(report_dir, 'DoD_filt_mean_morph_quant.txt'), DoD_filt_mean_morph_quant, header='tot_vol, sum_vol, dep_vol, sco_vol, morph_act_area, morph_act_area_dep, morph_act_area_sco, act_width_mean, act_width_mean_dep, act_width_mean_sco, act_thickness, act_thickness_dep, act_thickness_sco, bri')
        np.savetxt(os.path.join(report_dir, 'DoD_filt_isol_morph_quant.txt'), DoD_filt_isol_morph_quant, header='tot_vol, sum_vol, dep_vol, sco_vol, morph_act_area, morph_act_area_dep, morph_act_area_sco, act_width_mean, act_width_mean_dep, act_width_mean_sco, act_thickness, act_thickness_dep, act_thickness_sco, bri')
        np.savetxt(os.path.join(report_dir, 'DoD_filt_fill_morph_quant.txt'), DoD_filt_fill_morph_quant, header='tot_vol, sum_vol, dep_vol, sco_vol, morph_act_area, morph_act_area_dep, morph_act_area_sco, act_width_mean, act_width_mean_dep, act_width_mean_sco, act_thickness, act_thickness_dep, act_thickness_sco, bri')
        np.savetxt(os.path.join(report_dir, 'DoD_filt_nature_morph_quant.txt'), DoD_filt_nature_morph_quant, header='tot_vol, sum_vol, dep_vol, sco_vol, morph_act_area, morph_act_area_dep, morph_act_area_sco, act_width_mean, act_width_mean_dep, act_width_mean_sco, act_thickness, act_thickness_dep, act_thickness_sco, bri')
        np.savetxt(os.path.join(report_dir, 'DoD_filt_isol2_morph_quant.txt'), DoD_filt_isol2_morph_quant, header='tot_vol, sum_vol, dep_vol, sco_vol, morph_act_area, morph_act_area_dep, morph_act_area_sco, act_width_mean, act_width_mean_dep, act_width_mean_sco, act_thickness, act_thickness_dep, act_thickness_sco, bri')
        np.savetxt(os.path.join(report_dir, 'DoD_filt_ult_morph_quant.txt'), DoD_filt_ult_morph_quant, header='tot_vol, sum_vol, dep_vol, sco_vol, morph_act_area, morph_act_area_dep, morph_act_area_sco, act_width_mean, act_width_mean_dep, act_width_mean_sco, act_thickness, act_thickness_dep, act_thickness_sco, bri')
        
DoD_length_array = np.append(DoD_length_array, DoD_length)


# if run_mode==2:
#     # Print vulume teporal scale report
#     volume_temp_scale_report_header = 'run name, B_dep [min], SD(B_dep) [min], B_sco [min], SD(B_sco) [min]'
#     # Write temporl scale report as:
#     # run name, B_dep, SD(B_dep), B_sco, SD(B_sco)
#     with open(os.path.join(report_dir, 'volume_temp_scale_report.txt'), 'w') as fp:
#         fp.write(volume_temp_scale_report_header)
#         fp.writelines(['\n'])
#         for i in range(0,len(RUNS)):
#             for j in range(0, volume_temp_scale_report.shape[1]+1):
#                 if j == 0:
#                     fp.writelines([RUNS[i]+', '])
#                 else:
#                     fp.writelines(["%.3f, " % float(volume_temp_scale_report[i,j-1])])
#             fp.writelines(['\n'])
#     fp.close()

end = time.time()
print()
print('Execution time: ', (end-start), 's')
