'''
The aim of this script is to generate envelopes from stack

'''

import numpy as np
import os
from PIL import Image
import scipy.ndimage
from functions import *




# -------------------------------- DEFINE PATH ------------------------------- #
home_dir = os.getcwd()  # Home directory
DoDs_stack_folder = os.path.join(home_dir, 'DoDs', 'DoDs_stack')  # Input folder
PiQs_folder_path = '/home/erri/Documents/PhD/Research/5_research_repos/PiQs_analysis/'

set_name = 'q07_1'
run = 'q07r1'
DoD_time_span = 0

# ---------------------- DEFINE THE INPUT DATA TIMESCALE --------------------- #
# env_tscale_array = [5, 5, 5, 5]
env_tscale_array = [1,1,1,1]
# env_tscale_array = [5,7,12,18]
# env_tscale_array = [10,14,24,36]
# ---------------------------- ENVELOPE TIMESCALE ---------------------------- #
if run[1:3] == '07':
    env_tscale = env_tscale_array[3]
if run[1:3] == '10':
    env_tscale = env_tscale_array[2]
if run[1:3] == '15':
    env_tscale = env_tscale_array[1]
if run[1:3] == '20':
    env_tscale = env_tscale_array[0]

DoD_stack_name = 'DoD_stack_bool_' + set_name + '.npy'

DoD_stack = np.load(os.path.join(DoDs_stack_folder, DoD_stack_name))

DoD_stack = DoD_stack[DoD_time_span,:,:,:]

DoD_stack = np.where(DoD_stack == -1, 1, DoD_stack)

DoD_stack_envelope = np.nansum(DoD_stack, axis=2)


# RESCALE THE MATRIX:
DoD_stack_envelope_plot = np.where(np.isnan(DoD_stack_envelope), 0, DoD_stack_envelope)
DoD_stack_envelope_rsz = np.repeat(DoD_stack_envelope_plot, 10, axis=1) # Rescale the envMAA (dx/dy = 10) (at this stage 144x2790)
# DoD_stack_envelope_rsz = np.round(DoD_stack_envelope_rsz, decimals=2)
# SAVE
np.savetxt(os.path.join(DoDs_stack_folder, set_name + '_timespan' + str(DoD_time_span) +
           '_envMAA_history.txt'), DoD_stack_envelope_rsz, fmt='%.2f', delimiter='\t')



DoD_stack_envelope_bool = np.where(DoD_stack_envelope>0, 1, DoD_stack_envelope)

# RESCALE THE MATRIX:
DoD_stack_envelope_bool_plot = np.where(np.isnan(DoD_stack_envelope_bool), 0, DoD_stack_envelope_bool)
DoD_stack_envelope_bool_rsz = np.repeat(DoD_stack_envelope_bool_plot, 10, axis=1) # Rescale the envMAA (dx/dy = 10) (at this stage 144x2790)

# DoD_stack_envelope_bool_rsz = np.round(DoD_stack_envelope_bool_rsz, decimals=2)

# ---------------------- CUT LASER OUTPUT TO FIT PHOTOS ---------------------- #
DoD_stack_envelope_rsz = DoD_stack_envelope_rsz[:, DoD_stack_envelope_rsz.shape[1]-1229:]
DoD_stack_envelope_bool_rsz = DoD_stack_envelope_bool_rsz[:, DoD_stack_envelope_bool_rsz.shape[1]-1229:]

# mask_arr_rsz = mask_arr_rsz[:, mask_arr_rsz.shape[1]-1229:]
# DEM_rsz = DEM_rsz*mask_arr_rsz # Apply mask
# SAVE
np.savetxt(os.path.join(DoDs_stack_folder, set_name + '_timespan' + str(DoD_time_span) + '_envMAA_bool.txt'), DoD_stack_envelope_bool_rsz, fmt='%.2f', delimiter='\t')


if set_name == 'q07_1':
    # Define the transformation parameters
    scale = 1  # Enlargement scale
    DoD_scale = 1
    dx = 0  # Shift in x direction
    DoD_dx = 0
    dy = 0  # Shift in y direction
    DoD_dy = 0
    rot_angle = -0.55

if set_name == 'q10_2':
    # Define the transformation parameters
    scale = 1.0  # Enlargement scale
    dx = 0  # Shift in x direction
    dy = 8  # Shift in y direction
    rot_angle = -0.3

if set_name == 'q15_2':
    # Define the transformation parameters
    scale = 1.0  # Enlargement scale
    dx = 0  # Shift in x direction
    dy = 8  # Shift in y direction
    rot_angle = -0.4

if set_name == 'q20_2':
    # Define the transformation parameters
    scale = 1.0  # Enlargement scale
    dx = -10  # Shift in x direction
    dy = 8 # Shift in y direction
    rot_angle = -0.3
    
    



DoD_stack_envelope_bool_rsz_img = img_scaling_to_DEM(DoD_stack_envelope_bool_rsz, DoD_scale, DoD_dx, DoD_dy, rot_angle)

DoD_stack_envelope_bool_rsz_img = np.where(DoD_stack_envelope_bool_rsz_img>0, 255, DoD_stack_envelope_bool_rsz_img)
DoD_stack_envelope_bool_rsz_img = Image.fromarray(np.array(DoD_stack_envelope_bool_rsz_img).astype(np.uint8))
DoD_stack_envelope_bool_rsz_img.save(os.path.join(DoDs_stack_folder, run + '_envMAA_img.tif'))





'''The stack bool diff comes from the difference of maps taken at a given timescale'''
path_partial_envelopes = os.path.join(PiQs_folder_path, 'output_report/partial_envelopes', run + '_envTscale' + str(env_tscale))
stack_bool_diff_cld_path = os.path.join(path_partial_envelopes, run + '_envT' + str(env_tscale) + '_partial_stack_bool_diff_cld.npy')
stack_bool_diff_cld = np.load(stack_bool_diff_cld_path)

activated_pixel_envelope = np.nansum(stack_bool_diff_cld == +1, axis=0)
deactivated_pixel_envelope = np.nansum(stack_bool_diff_cld == -1, axis=0)

# # Resizing:
# # Order has to be 0 to avoid negative numbers in the cumulate intensity.
# resampling_factor = 5

# activated_pixel_envelope_rsz = scipy.ndimage.zoom(activated_pixel_envelope, 1/resampling_factor, mode='nearest', order=1)
# activated_pixel_envelope_rsz = np.repeat(activated_pixel_envelope_rsz, 10, axis=1)  # Rescale the DEM (dx/dy = 10)

# deactivated_pixel_envelope_rsz = scipy.ndimage.zoom(deactivated_pixel_envelope, 1/resampling_factor, mode='nearest', order=1)
# deactivated_pixel_envelope_rsz = np.repeat(deactivated_pixel_envelope_rsz, 10, axis=1)  # Rescale the DEM (dx/dy = 10)


# Convert image in  uint16 (unsigned 16-bit integer)
activated_pixel_envelope_rsz = activated_pixel_envelope
deactivated_pixel_envelope_rsz = deactivated_pixel_envelope


# Convert image in  uint16 (unsigned 16-bit integer)
activated_pixel_envelope_rsz = activated_pixel_envelope_rsz.astype(np.uint16)
deactivated_pixel_envelope_rsz = deactivated_pixel_envelope_rsz.astype(np.uint16)

#%%



activated_pixel_envelope_rsz_rsc = img_scaling_to_DEM(activated_pixel_envelope_rsz, scale, dx, dy, rot_angle)
deactivated_pixel_envelope_rsz_rsc = img_scaling_to_DEM(deactivated_pixel_envelope_rsz, scale, dx, dy, rot_angle)


# Convert and save

deactivated_pixel_envelope_rsz_rsc = np.where(deactivated_pixel_envelope_rsz_rsc>0, 255, deactivated_pixel_envelope_rsz_rsc)
deactivated_pixel_envelope_rsz_rsc_img = Image.fromarray(np.array(deactivated_pixel_envelope_rsz_rsc).astype(np.uint8))
deactivated_pixel_envelope_rsz_rsc_img.save(os.path.join(DoDs_stack_folder, run + '_deact_scaled_img.tif'))
# Convert and save
activated_pixel_envelope_rsz_rsc = np.where(activated_pixel_envelope_rsz_rsc>0, 255, activated_pixel_envelope_rsz_rsc)
activated_pixel_envelope_rsz_rsc_img = Image.fromarray(np.array(activated_pixel_envelope_rsz_rsc).astype(np.uint8))
activated_pixel_envelope_rsz_rsc_img.save(os.path.join(DoDs_stack_folder, run + '_act_scaled_img.tif'))

print()



