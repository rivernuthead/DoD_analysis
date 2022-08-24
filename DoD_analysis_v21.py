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
from mpl_toolkits.axes_grid1.inset_locator import mark_inset, inset_axes
# from matplotlib.colors import ListedColormap, BoundaryNorm

start = time.time() # Set initial time

###############################################################################
# SETUP SCRIPT PARAMETERS and RUN MODE
###############################################################################


'''
run mode:
    1 = one run at time
    2 = bath process
DEM analysis mode:
    0 = do not perform DEM analysis
    1 = perform DEM analysis
data_interpolatuon_mode:
    0 = no interpolation
    1 = data interpolation
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
DEM_analysis_mode = 1
data_interpolation_mode = 1
mask_mode = 1
process_mode = 1
save_plot_mode = 1

# SINGLE RUN NAME
run = 'q10_3'

# Set DEM single name to perform process to specific DEM
DEM1_single_name = 'matrix_bed_norm_q07S0.txt' # DEM1 name
DEM2_single_name = 'matrix_bed_norm_q07S1.txt' # DEM2 name

# Filtering process thresholds values
thrs_1 = 2.0  # [mm] # Lower threshold
thrs_2 = 15.0  # [mm] # Upper threshold
neigh_thrs = 5  # [-] # Number of neighborhood cells for validation

# Survey pixel dimension
px_x = 50 # [mm]
px_y = 5 # [mm]

# Not a number raster value (NaN)
NaN = -999

# Engelund-Gauss model parameters
g = 9.806 # Gravity
ds = 0.001  # Sediment grainsize [mm]
teta_c = 0.02 # Schield parameter [-]
NG=4 # Number of Gauss points
max_iter = 100000 # Maximum numer of iterations
toll = 0.00001 # Convergence tolerance

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
    
    files=[] # initializing filenames list
    # Creating array with file names:
    for f in sorted(os.listdir(input_dir)):
        path = os.path.join(input_dir, f)
        if os.path.isfile(path) and f.endswith('.txt') and f.startswith('matrix_bed_norm_'+run+'s'):
            files = np.append(files, f)

    # INITIALIZE ARRAYS
    comb = np.array([]) # combination of differences
    DoD_count_array=[] # Active pixel
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
    # DEM ANALYSIS
    ###########################################################################
    if DEM_analysis_mode==1:
        # - Residual slope, for each DEM
        # - Bed Relief Index (BRI) averaged, for each DEM
        # - STDEV (SD) of the bed elevation, for each DEM
        # Initialize arrays
        slope_res = [] # Rsidual slope array
        BRI=[] # BRi array
        SD = [] # SD array
        engelund_model_array=[] # Engelund model array (Q, D, Wwet/w])
        water_depth_array=[] # Water dept array [m]
        discharge_array=[] # Discarge [m^3/s]
        Wwet_array = [] # Wwet array [Wwet/W]
        # morphWact_values = [] # All the morphological active width values for each runs

        for f in files:
            DEM_path = os.path.join(input_dir, f) # Set DEM path
            DEM = np.loadtxt(DEM_path,          # Load DEM data
                             #delimiter=',',
                             skiprows=8)
            DEM = np.where(np.isclose(DEM, NaN), np.nan, DEM)

            # DEM reshaping according to arr_shape...
            DEM=DEM[0:arr_shape[0], 0:arr_shape[1]]

            # DEM masking...
            DEM = DEM*array_mask_rshp_nan

            # Residual slope
            # NB: this operation will be performed to detrended DEMs
            # Averaged crosswise bed elevation array:
            bed_profile = np.nanmean(DEM, axis=0) # Bed profile
            # Linear regression of bed profile:
            # Performing linear regression
            x_coord = np.linspace(0, px_x*len(bed_profile), len(bed_profile)) # Longitudinal coordinate
            linear_model = np.polyfit(x_coord, bed_profile,1) # linear_model[0]=m, linear_model[1]=q y=m*x+q
            slope_res = np.append(slope_res, linear_model[0]) # Append residual slope values

            # PLOT cross section mean values and trendline
            # fig, ax1 = plt.subplots(dpi=200)
            # ax1.plot(x_coord, bed_profile)
            # ax1.plot(x_coord, x_coord*linear_model[0]+linear_model[1], color='red')
            # ax1.set(xlabel='longitudinal coordinate (mm)', ylabel='Z (mm)',
            #        title=run+'\n'+'Residual slope:'+str(linear_model[0]))

            # BRI calculation
            BRI=np.append(BRI,np.mean(np.nanstd(DEM, axis=0))) # Overall BRI
            
            if f == files[0]:
                BRI_array = np.nanstd(DEM, axis=0) # Array crosswise for each DEM
            else:
                BRI_array = np.vstack((BRI_array, np.nanstd(DEM, axis=0)))

            # Bed elevation STDEV
            SD = np.append(SD,np.nanstd(DEM))

            # Create report matrix:
            # Structure: DEM name, residual slope [m/m], BRI [mm], SD [mm]
            matrix_DEM_analysis = np.transpose(np.stack((slope_res, BRI, SD)))

            # Build report
            report_DEM_header = 'DEM name, residual slope [m/m], BRI [mm], SD [mm]'
            report_DEM_name = run+'_DEM_report.txt'
            with open(os.path.join(report_dir, report_DEM_name), 'w') as fp:
                fp.write(report_DEM_header)
                fp.writelines(['\n'])
                for i in range(0,len(matrix_DEM_analysis[:,0])):
                    for j in range(0, len(matrix_DEM_analysis[0,:])+1):
                        if j == 0:
                            fp.writelines([files[i]+', '])
                        elif j==1:
                            # fp.writelines(["%.6f, " % float(matrix_DEM_analysis[i,j-1])])
                            fp.writelines(["{:e},".format(matrix_DEM_analysis[i,j-1])])
                        else:
                            fp.writelines(["%.3f, " % float(matrix_DEM_analysis[i,j-1])])
                    fp.writelines(['\n'])
            fp.close()

            # DEM detrending (DEM detrended both with slope and residual slope)
            DEM_detrended = DEM
            for i in range(0,DEM.shape[1]):
                DEM_detrended[:,i] = DEM[:,i]-linear_model[0]*i*px_x

#%%
            # Create equivalent cross section as sorted DEM vaues excluding NaN
            DEM_values = sorted(DEM_detrended[np.logical_not(np.isnan(DEM_detrended))]) # array with DEM values
            # cross_section_eq = DEM_values[::100] # Resize DEM value to be lighter (100 res resampling)
            cross_section_eq = np.interp(np.arange(0,len(DEM_values),len(DEM_values)/W/1000), np.arange(0,len(DEM_values)), DEM_values)
            # Add cross section banks as the double of the maximum DEM's value:
            z_coord = np.pad(cross_section_eq, (1,1), mode='constant', constant_values=int(cross_section_eq.max()*2))
            z_coord = z_coord/1000 # Convert z_coord in meters
#%%
            # Create cross-wise coordination
            y_coord = np.arange(0,W*1000, W*1000/len(z_coord))
            y_coord = y_coord/1000 # Convert y_coord in meters

            # ENGENLUND-GAUSS IMPLEMENTATION

            Dmax = z_coord.max()-z_coord.min() # Maximum water dept
            Dmin = 0 # Minimum water level
            i=0 # Initialize iteration counter

            # Guess values:
            D0 = (Dmax-Dmin)/2 # Water dept
            Qn, Omega, b, B, alpha, beta, Qs, count_active = MotoUniforme(S, y_coord, z_coord, D0, NG, teta_c, ds) # Discharge
            # Discharge extreme values
            Qmax, Omega, b, B, alpha, beta, Qs, count_active = MotoUniforme(S, y_coord, z_coord, Dmax, NG, teta_c, ds)
            Qmin, Omega, b, B, alpha, beta, Qs, count_active = MotoUniforme(S, y_coord, z_coord, Dmin, NG, teta_c, ds)
            Q_target = Q/1000 # Target discharge [m^3/s]
            if np.sign(Qmax-Q_target)==np.sign(Qmin-Q_target):
                print(' Soluntion out of boundaries')
            else:
                # Check if h<h_min:
                while abs(Qn - Q_target)>toll:
                    if i>max_iter:
                        print('ERROR: max iterations reached!')
                        break
                    i+=1
                    D0 = (Dmax+Dmin)/2
                    Q0, Omega, b, B, alpha, beta, Qs, count_active = MotoUniforme(S, y_coord, z_coord, D0, NG, teta_c, ds)
                    if Q0>Q_target:
                        Dmax=D0 # Update Dmax
                    elif Q0<Q_target:
                        Dmin=D0 # Update Dmin
                    Qn=Q0

            water_depth_array=np.append(water_depth_array, D0) # Water depth array
            discharge_array=np.append(discharge_array, Q0) # Discarge
            Wwet_array = np.append(Wwet_array, b/W)

        # BRI plot
        #TODO
        n_data = np.linspace(0,len(files)-1,len(files)) # Linspace of the number of available DoD
        c_data = np.linspace(0,1,len(files)) # c_data needs to be within 0 and 1
        colors = plt.cm.viridis(c_data)
        fig1, axs = plt.subplots(1,1,dpi=400, sharex=True, tight_layout=True, figsize=(8,6))
        #Defines the size of the zoom window and the positioning
        axins = inset_axes(axs, 2, 5, loc = 1, bbox_to_anchor=(1.3, 0.9),
                            bbox_transform = axs.figure.transFigure)
        for d, color in zip(n_data, colors):
            DoD_name = files[int(d)]
            axs.plot(np.linspace(0,DEM.shape[1]-1, DEM.shape[1])*px_x/1000, BRI_array[int(d),:], '-', c=color, label=DoD_name[21:23])
            plt.plot(np.linspace(0,DEM.shape[1]-1, DEM.shape[1])*px_x/1000, BRI_array[int(d),:], '-', c=color, label=DoD_name[21:23])
        
        # axins.scatter(x, y)
        x1, x2 = 12, 14
        y1, y2 = np.min(BRI_array[:,int(x1/px_x*1000):])*0.9, np.max(BRI_array[:,int(x1/px_x*1000):])*1.1 #Setting the limit of x and y direction to define which portion to #zoom
        axins.set_xlim(x1, x2)
        axins.set_ylim(y1, y2)
        #Draw the lines from the portion to zoom and the zoom window
        mark_inset(axs, axins, loc1=1, loc2=3, fc="none", ec = "0.4")
        axs.set_title(run + ' - BRI', fontsize=14)
        axs.set_xlabel('Longitudinal coordinate [m]', fontsize=12)
        axs.set_ylabel('BRI', fontsize=12)
        # plt.savefig(os.path.join(plot_dir, run +'_morphW_interp.png'), dpi=200)
        plt.legend(loc='best', fontsize=8)
        
        # ax_new = fig1.add_axes([0.2, 1.1, 0.4, 0.4])
        # plt.plot(np.linspace(0,array.shape[1]-1, array.shape[1])*px_x, cross_bri_matrix[int(d),:], '-', c=color)
        
        plt.show()

        # BRI errorbar plot
        fig1, axs = plt.subplots(1,1,dpi=200, sharex=True, tight_layout=True, figsize=(8,6))
        axs.errorbar(np.linspace(0,DEM.shape[1]-1, DEM.shape[1])*px_x/1000, np.nanmean(BRI_array, axis=0), np.std(BRI_array, axis=0), linestyle='--', marker='^', color='darkcyan')
        axs.tick_params(axis='y', labelcolor='darkcyan')
        axs.set_title(run, fontsize=14)
        # axs.set_xlim(0, np.max(mean_matrix[1,:]*px_x)+1)
        axs.set_xlabel('Window analysis length [m]', fontsize=12)
        axs.set_ylabel('BRI [mm]', fontsize=12)
        # plt.savefig(os.path.join(plot_dir, run +'_morphW_interp.png'), dpi=200)
        plt.show()


        water_dept=np.mean(water_depth_array) # Average water dept
        discharge=np.mean(discharge_array) # Average discarge
        Wwet = np.mean(Wwet_array)
        print('Engelund-Gauss model results:')
        print('Reached discharge: ', discharge, ' m^3/s')
        print('Water dept: ', water_dept, ' m')
        print('Wwet/W: ', Wwet)

        # Append values as: run name, D [m], Q [m^3/s], Wwet/W [-]
        engelund_model_array = np.append(engelund_model_array,(water_dept, discharge, Wwet))
        if run_mode ==2:
            engelund_model_report[int(np.where(RUNS==run)[0]),:]=engelund_model_array

        # Print averaged residual slope:
        print()
        print('Averaged DEMs residual slope: ', np.average(slope_res))


    ###########################################################################
    # LOOP OVER ALL DEMs COMBINATIONS
    ###########################################################################
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
            DoD_name = 'DoD_' + DEM2_name[-6:-4] + '-' + DEM1_name[-6:-4] + '_'

            # Setup output folder
            output_name = 'script_outputs_' + DEM2_name[20:21] + '-' + DEM1_name[20:21] # Set outputs name


            path_out = os.path.join(home_dir, 'DoDs', 'DoD_'+run) # Set DoD outputs directory
            if not(os.path.exists(path_out)):
                os.mkdir(path_out)


            ###################################################################
            # DATA READING...
            ###################################################################
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

            
            ###################################################################
            # PERFORM DEM OF DIFFERENCE - DEM2-DEM1
            ###################################################################
            # Print DoD name
            print(DEM2_name, '-', DEM1_name)
            # Raster dimension
            dim_y, dim_x = DEM1.shape
            # dim_x, dim_x = DEM1.shape
            
            DoD_length = DEM1.shape[1]*px_x/1000 # DoD length [m]
            
            # DoD_length_array = np.append(DoD_length_array, DoD_length)

            # Creating DoD array with np.nan
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
            DoD_count_array = np.append(DoD_count_array, DoD_count)

            # DoD statistics
            # print('The minimum DoD value is:\n', np.nanmin(DoD_raw))
            # print('The maximum DoD value is:\n', np.nanmax(DoD_raw))
            # print('The DoD shape is:\n', DoD_raw.shape)

            ###################################################################
            # DATA FILTERING...
            ###################################################################

            # Perform domain-wide average
            domain_avg = np.pad(DoD_raw, 1, mode='edge') # i size pad with edge values domain
            DoD_mean = np.zeros(DEM1.shape)
            for i in range (0, dim_y):
                for j in range (0, dim_x):
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
            # # Filtered array weighted average by nan.array mask
            # DoD_mean = DoD_mean * array_msk_nan
            # Create a GIS readable DoD mean (np.nan as -999)
            DoD_mean = np.round(DoD_mean, 1) # Round data to 1 decimal precision
            DoD_mean_rst = np.where(np.isnan(DoD_mean), NaN, DoD_mean)

            # Threshold and Neighbourhood analysis process
            DoD_filt = np.copy(DoD_mean) # Initialize filtered DoD array as a copy of the averaged one
            DoD_filt_domain = np.pad(DoD_filt, 1, mode='edge') # Create neighbourhood analysis domain

            for i in range(0,dim_y):
                for j in range(0,dim_x):
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
            for i in range(0,dim_y):
                for j in range(0,dim_x):
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

            ###################################################################
            # PLOT RAW DOD, MEAN DOD AND FILTERED DOD
            ###################################################################
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

            ###################################################################
            # TOTAL VOLUMES, DEPOSITION VOLUMES AND SCOUR VOLUMES
            ###################################################################
            # TODO implement morph_quantities_func_v2.py
            # DoD filtered name: DoD_filt
            # Create new raster where apply volume calculation
            # DoD>0 --> Deposition, DoD<0 --> Scour
            # =+SUMIFS(A1:JS144, A1:JS144,">0")*5*50(LibreCalc function)
            
            # Define total volume matrix, Deposition matrix and Scour matrix
            DoD_vol = np.where(np.isnan(DoD_filt_nozero), 0, DoD_filt_nozero) # Total volume matrix
            dep_DoD = (DoD_vol>0)*DoD_vol # DoD of only deposition data
            sco_DoD = (DoD_vol<0)*DoD_vol # DoD of only scour data
            
            
            tot_vol = np.sum(DoD_vol)*px_x*px_y/(W*DoD_length*d50*1e09) # Total volume as V/(L*W*d50) [-] considering negative sign for scour
            sum_vol = np.sum(np.abs(DoD_vol))*px_x*px_y/(W*DoD_length*d50*1e09) # Sum of scour and deposition volume as V/(L*W*d50) [-]
            dep_vol = np.sum(dep_DoD)*px_x*px_y/(W*DoD_length*d50*1e09) # Deposition volume as V/(L*W*d50) [-]
            sco_vol = np.sum(sco_DoD)*px_x*px_y/(W*DoD_length*d50*1e09) # Scour volume as V/(L*W*d50) [-]
            
            
            #Print results:
            print('Total volume V/(L*W*d50) [-]:', "{:.1f}".format(tot_vol))
            print('Sum of deposition and scour volume V/(L*W*d50) [-]:', "{:.1f}".format(sum_vol))
            print('Deposition volume V/(L*W*d50) [-]:', "{:.1f}".format(dep_vol))
            print('Scour volume V/(L*W*d50) [-]:', "{:.1f}".format(sco_vol))

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
            
            print('Active thickness [mm]:', act_thickness)
            print('Morphological active area: ', "{:.1f}".format(morph_act_area), '[mm²]')
            print('Morphological active width (mean):', "{:.3f}".format(act_width_mean), '%')
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

            DEM1_num=DEM1_name[-5:-4] # DEM1 number
            DEM2_num=DEM2_name[-5:-40] # DEM2 number
            delta=int(DEM2_name[-5:-4])-int(DEM1_name[-5:-4]) # Calculate delta between DEM

            # Build up morphWact/W array for the current run boxplot
            # This array contain all the morphWact/W values for all the run repetition in the same line
            # This array contain only adjacent DEMs DoD
            if delta==1:
                morphWact_values = np.append(morphWact_values, act_width_array)

            # Fill Scour, Deposition and morphWact/w matrix:
            if delta != 0:
                # Fill matrix with values
                matrix_volumes[delta-1,h]=np.sum(DoD_vol)*px_x*px_y/(W*DoD_length*d50*1e09) # Total volumes as the algebric sum of scour and deposition volumes V/(W*L) [mm]
                matrix_sum_volumes[delta-1,h]=np.sum(np.abs(DoD_vol))*px_x*px_y/(W*DoD_length*d50*1e09) # Total volumes as the sum of scour and deposition volumes V/(W*L) [mm]
                matrix_dep[delta-1,h]=np.sum(dep_DoD)*px_x*px_y/(W*DoD_length*d50*1e09) # Deposition volumes as V/(W*L) [mm]
                matrix_sco[delta-1,h]=np.sum(sco_DoD)*px_x*px_y/(W*DoD_length*d50*1e09) # Scour volumes as V/(W*L) [mm]
                matrix_morph_act_area[delta-1,h]=morph_act_area # Total morphological active area
                matrix_morph_act_area_dep[delta-1,h]=morph_act_area_dep # Deposition morphological active area
                matrix_morph_act_area_sco[delta-1,h]=morph_act_area_sco # Scour morphological active area

                # Fill last two columns with AVERAGE and STDEV
                matrix_volumes[delta-1,-2]=np.average(matrix_volumes[delta-1,:len(files)-delta]) #Total volumes
                matrix_sum_volumes[delta-1,-2]=np.average(matrix_sum_volumes[delta-1,:len(files)-delta]) #Total sum volumes
                matrix_dep[delta-1,-2]=np.average(matrix_dep[delta-1,:len(files)-delta]) # Deposition volumes
                matrix_sco[delta-1,-2]=np.average(matrix_sco[delta-1,:len(files)-delta]) # Scour volumes
                matrix_morph_act_area[delta-1,-2]=np.average(matrix_morph_act_area[delta-1,:len(files)-delta]) # Morphological total active area
                matrix_morph_act_area_dep[delta-1,-2]=np.average(matrix_morph_act_area_dep[delta-1,:len(files)-delta]) # Morphological deposition active area
                matrix_morph_act_area_sco[delta-1,-2]=np.average(matrix_morph_act_area_sco[delta-1,:len(files)-delta]) # Morphological scour active area
                
                matrix_volumes[delta-1,-1]=np.std(matrix_volumes[delta-1,:len(files)-delta])
                matrix_sum_volumes[delta-1,-1]=np.std(matrix_sum_volumes[delta-1,:len(files)-delta])
                matrix_dep[delta-1,-1]=np.std(matrix_dep[delta-1,:len(files)-delta])
                matrix_sco[delta-1,-1]=np.std(matrix_sco[delta-1,:len(files)-delta])
                matrix_morph_act_area[delta-1,-1]=np.std(matrix_morph_act_area[delta-1,:len(files)-delta])
                matrix_morph_act_area_dep[delta-1,-1]=np.std(matrix_morph_act_area_dep[delta-1,:len(files)-delta])
                matrix_morph_act_area_sco[delta-1,-1]=np.std(matrix_morph_act_area_sco[delta-1,:len(files)-delta])

                # Fill active thickness matrix:
                matrix_act_thickness[delta-1,h]=act_thickness #Fill matrix with active thickness data calculated from total volume matrix
                matrix_act_thickness_dep[delta-1,h]=act_thickness_dep #Fill matrix with active thickness data calculated from deposition volume matrix
                matrix_act_thickness_sco[delta-1,h]=act_thickness_sco #Fill matrix with active thickness data calculated from scour volume matrix
                
                matrix_act_thickness[delta-1,-2]=np.average(matrix_act_thickness[delta-1,:len(files)-delta]) # Fill matrix with active thickness average calculated from total volume matrix
                matrix_act_thickness_dep[delta-1,-2]=np.average(matrix_act_thickness_dep[delta-1,:len(files)-delta]) # Fill matrix with active thickness average calculated from deposition volume matrix
                matrix_act_thickness_sco[delta-1,-2]=np.average(matrix_act_thickness_sco[delta-1,:len(files)-delta]) # Fill matrix with active thickness average calculated from scour volume matrix
                
                matrix_act_thickness[delta-1,-1]=np.std(matrix_act_thickness[delta-1,:len(files)-delta]) # Fill matrix with active thickness standard deviation calculated from total volume matrix
                matrix_act_thickness_dep[delta-1,-1]=np.std(matrix_act_thickness_dep[delta-1,:len(files)-delta]) # Fill matrix with active thickness average calculated from deposition volume matrix
                matrix_act_thickness_sco[delta-1,-1]=np.std(matrix_act_thickness_sco[delta-1,:len(files)-delta]) # Fill matrix with active thickness average calculated from scour volume matrix

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

                matrix_Wact[delta-1,h]=act_width_mean
                matrix_Wact[delta-1,-2]=np.average(matrix_Wact[delta-1,:len(files)-delta])
                matrix_Wact[delta-1,-1]=np.std(matrix_Wact[delta-1,:len(files)-delta])


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

                # Fill III quantile Wact/W matrix:
                matrix_Wact_IIIquantile[delta-1,h]=np.quantile(act_width_array, .75)
                matrix_Wact_IIIquantile[delta-1,-2]=np.min(matrix_Wact_IIIquantile[delta-1,:len(files)-delta])
                matrix_Wact_IIIquantile[delta-1,-1]=np.max(matrix_Wact_IIIquantile[delta-1,:len(files)-delta])

                # Fill I quantile Wact/W matrix:
                matrix_Wact_Iquantile[delta-1,h]=np.quantile(act_width_array, .25)
                matrix_Wact_Iquantile[delta-1,-2]=np.min(matrix_Wact_Iquantile[delta-1,:len(files)-delta])
                matrix_Wact_Iquantile[delta-1,-1]=np.max(matrix_Wact_Iquantile[delta-1,:len(files)-delta])                
                
                
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
                DoD_stack1[h,:,:] = DoD_filt_nozero_rst[:,:]

            ###################################################################
            # SAVE DATA
            ###################################################################

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

            # ACTIVE PIXEL DoD
            # Print boolean map of active pixel: 1=active, 0=not active
            np.savetxt(path_out + '/' + DoD_name + 'active.txt', act_px_matrix, fmt='%0.1f', delimiter='\t')

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
    
    ###################################################################
    # 3D ARRAY ANALISYS
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

    # Print the last DoD outcome
    if save_plot_mode == 1:
        fig, ax = plt.subplots(dpi=200, tight_layout=True)
        im = ax.imshow(np.where(DoD_filt_nozero_rst==NaN, np.nan, DoD_filt_nozero_rst), cmap='RdBu',  vmin=-25, vmax=25, aspect='0.1')
        plt.colorbar(im)
        plt.title(DoD_name[:-1], fontweight='bold')
        # plt.savefig(os.path.join(plot_dir, run +'_DoD.png'), dpi=200)
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
            font = {'family': 'serif',
                    'color':  'black',
                    'weight': 'regular',
                    'size': 10
                    }
            fig1, axs = plt.subplots(2,1,dpi=200, sharex=True, tight_layout=True)
            axs[0].plot(xData, yData_dep, 'o')
            axs[0].plot(xData, intCurve_dep, c='red')
            axs[0].set_title('Deposition volumes interpolation '+run)
            axs[0].set_xlabel('Time [min]')
            axs[0].set_ylabel('Volume V/(L*W) [mm]')
            axs[0].text(np.max(xData)*0.7, np.min(yData_dep), 'Trun=' + str(dt) + 'min \n' + r'$\tau$=' + str(np.round(par_dep[1], decimals=1)) + 'min \n' + 'A = ' + str(np.round(par_dep[0], decimals=1)), fontdict=font)
            axs[1].plot(xData, yData_sco, 'o')
            axs[1].plot(xData, intCurve_sco, c='red')
            axs[1].set_title('Scour volumes interpolation '+run)
            axs[1].set_xlabel('Time [min]')
            axs[1].set_ylabel('Volume V/(L*W) [mm]')
            axs[1].text(np.max(xData)*0.7, np.min(yData_sco), 'Trun=' + str(dt) + 'min \n' + r'$\tau$=' + str(np.round(par_sco[1], decimals=1)) + 'min \n' + 'A = ' + str(np.round(par_sco[0], decimals=1)), fontdict=font)
            plt.savefig(os.path.join(plot_dir, run +'_volume_interp.png'), dpi=200)
            plt.show()
    
            fig2, axs = plt.subplots(1,1,dpi=200, sharex=True, tight_layout=True)
            axs.plot(xData, yData_morphW, 'o', c='brown')
            axs.plot(xData, intCurve_morphW, c='green')
            axs.set_title('Morphological active width (morphW/W) '+run)
            axs.set_xlabel('Time [min]')
            axs.set_ylabel('morphW/W [-]')
            plt.text(np.max(xData)*0.7, np.min(yData_morphW), 'Trun=' + str(dt) + 'min \n' + r'$\tau$=' + str(np.round(par_sco[1], decimals=1)) + 'min \n' + 'A = ' + str(np.round(par_sco[0], decimals=1)), fontdict=font)
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
    report_matrix = np.array(np.transpose(np.stack((comb, DoD_count_array, volumes_array, dep_array, sco_array, morph_act_area_array, act_width_mean_array))))
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
    np.savetxt(report_sum_vol_name, matrix_sum_volumes, fmt='%.5f', delimiter=',', newline='\n')
    
    # Create deposition matrix report
    report_dep_name = os.path.join(report_dir, run +'_dep_report.txt')
    np.savetxt(report_dep_name, matrix_dep, fmt='%.5f', delimiter=',', newline='\n')

    # Create scour matrix report
    report_sco_name = os.path.join(report_dir, run +'_sco_report.txt')
    np.savetxt(report_sco_name, matrix_sco, fmt='%.5f', delimiter=',', newline='\n')
    
    # Create total active thickness matrix report (calculated from volume matrix)
    report_act_thickness_name = os.path.join(report_dir, run +'_act_thickness_report.txt')
    np.savetxt(report_act_thickness_name, matrix_act_thickness , fmt='%.5f', delimiter=',', newline='\n')
    
    # Create deposition active thickness matrix report (calculated from deposition volume matrix)
    report_act_thickness_name_dep = os.path.join(report_dir, run +'_act_thickness_report_dep.txt')
    np.savetxt(report_act_thickness_name_dep, matrix_act_thickness_dep , fmt='%.5f', delimiter=',', newline='\n')
    
    # Create scour active thickness matrix report (calculated from scour volume matrix)
    report_act_thickness_name_sco = os.path.join(report_dir, run +'_act_thickness_report_sco.txt')
    np.savetxt(report_act_thickness_name_sco, matrix_act_thickness_sco , fmt='%.5f', delimiter=',', newline='\n')
    
    # Create total active area matrix report (calculated from volume matrix)
    report_act_area_name = os.path.join(report_dir, run + '_act_area_report.txt')
    np.savetxt(report_act_area_name, matrix_morph_act_area, fmt='%.5f', delimiter=',', newline='\n')
    
    # Create deposition active area matrix report (calculated from volume matrix)
    report_act_area_name_dep = os.path.join(report_dir, run + '_act_area_report_dep.txt')
    np.savetxt(report_act_area_name_dep, matrix_morph_act_area_dep, fmt='%.5f', delimiter=',', newline='\n')
    
    # Create scour active area matrix report (calculated from volume matrix)
    report_act_area_name_sco = os.path.join(report_dir, run + '_act_area_report_sco.txt')
    np.savetxt(report_act_area_name_sco, matrix_morph_act_area_sco, fmt='%.5f', delimiter=',', newline='\n')

    # Create Wact report matrix
    matrix_Wact=matrix_Wact[:len(files)-1,:] # Fill matrix_Wact with morphological  active width values
    matrix_Wact[:,len(files)-1]=matrix_Wact_Iquantile[:,len(files)-1] # Fill matrix_Wact report with minimum values
    matrix_Wact[:,len(files)]=matrix_Wact_IIIquantile[:,len(files)] # Fill matrix_Wact report with maximum values
    report_Wact_name = os.path.join(report_dir, run +'_morphWact_report.txt')
    np.savetxt(report_Wact_name, matrix_Wact, fmt='%.5f', delimiter=',', newline='\n')

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
                fp.writelines(["%.5f" % float(morphWact_values[i])])
            else:
                fp.writelines(["%.5f," % float(morphWact_values[i])])
        fp.writelines(['\n'])
    fp.close()



    ###########################################################################
    # PLOTS
    ###########################################################################
    # Define arrays for scour and volume data over time
    xData=np.arange(1, len(files), 1)*dt_xnr # Time in Txnr
    yData_sco=np.absolute(matrix_sco[:len(files)-1,0])
    yError_sco=matrix_sco[:len(files)-1,-1]
    yData_dep=np.absolute(matrix_dep[:len(files)-1,0])
    yError_dep=matrix_dep[:len(files)-1,-1]
    yData_act_thickness=matrix_act_thickness[:len(files)-1,0]
    yError_act_thickness=matrix_act_thickness[:len(files)-1,-1]

    if save_plot_mode==1:
        fig3, axs = plt.subplots(2,1,dpi=80, figsize=(10,6), sharex=True, tight_layout=True)
        fig3.suptitle(run + ' - Volume')
        axs[0].errorbar(xData,yData_sco, yError_sco, linestyle='--', marker='^', color='red')
        axs[0].set_ylim(bottom=0)
        axs[0].set_title('Scour')
        # axs[0].set_xlabel()
        axs[0].set_ylabel('Scour volume V/(L*W*d50) [-]')
        axs[1].errorbar(xData,yData_dep, yError_dep, linestyle='--', marker='^', color='blue')
        axs[1].set_ylim(bottom=0)
        axs[1].set_title('Deposition')
        axs[1].set_xlabel('Exner time')
        axs[1].set_ylabel('Scour olume V/(L*W*d50) [-]')
        plt.savefig(os.path.join(plot_dir, run +'dep_scour.png'), dpi=200)
        plt.show()
        
        
        fig4, axs = plt.subplots(1,1,dpi=80, figsize=(10,6), sharex=True, tight_layout=True)
        axs.errorbar(xData,yData_act_thickness, yError_act_thickness, linestyle='--', marker='^', color='purple')
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

    xData=np.arange(1, len(files), 1)*dt
    volume_over_time_matrix = []
    volume_over_time_matrix = np.stack((xData, yData_dep, -yData_sco))

    # Append rows to the current file
    with open(os.path.join(report_dir, 'volume_over_time.txt'), 'a') as fp:
        fp.writelines([run+', '])
        fp.writelines(['\n'])
        for i in range(0,volume_over_time_matrix.shape[0]):
            for j in range(0,volume_over_time_matrix.shape[1]):
                fp.writelines(["%.5f, " % float(volume_over_time_matrix[i,j])])
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
                    fp.writelines(["%.5f, " % float(volume_temp_scale_report[i,j-1])])
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
                    fp.writelines(["%.5f, " % float(morphW_temp_scale_report[i,j-1])])
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
                        fp.writelines(["%.5f, " % float(engelund_model_report[i,j-1])])
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

# Filter np.nan
fig, ax = plt.subplots(dpi=80, figsize=(10,6))
fig.suptitle('Dimensionless morphological active width', fontsize = 18)
for i in range(0, len(RUNS)):
    bplot=ax.boxplot(morphWact_matrix[i,:][~np.isnan(morphWact_matrix[i,:])], positions=[i], widths=0.5) # Data were filtered by np.nan values
ax.yaxis.grid(True)
ax.set_xlabel('Runs', fontsize=12)
ax.set_ylabel('morphWact/W [-]', fontsize=12)
plt.xticks(np.arange(0,len(RUNS), 1), RUNS)
plt.savefig(os.path.join(plot_dir, 'morphWact_boxplot.png'), dpi=200)
plt.show()



end = time.time()
print()
print('Execution time: ', (end-start), 's')
