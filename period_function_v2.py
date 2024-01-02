#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 15:47:48 2023

@author: erri
"""
import numpy as np


def find_consecutive_number_lengths(array, number):
    consecutive_number_lengths = []
    current_length = 0

    for num in array:
        if num == number:
            current_length += 1
        elif current_length > 0:
            consecutive_number_lengths.append(current_length)
            current_length = 0

    if current_length > 0:
        consecutive_number_lengths.append(current_length)

    return consecutive_number_lengths

#88888888888888888888888888888888888888888888888888888888888888888888888888888#
def trim_consecutive_equal(arr):
    
    arr = np.array(arr) # Make sure the array is a numpy array

    # Create an array that indicates whether each value is equal to the next
    equal_to_next = np.hstack((arr[:-1] == arr[1:], False))

    # Use this to mask out the consecutive equal values
    trimmed_arr = arr[~equal_to_next]

    return trimmed_arr
#-----------------------------------------------------------------------------#
#------------------------------------TEST-------------------------------------#


#%%888888888888888888888888888888888888888888888888888888888888888888888888888#
# def switch_counter(arr_raw):
    
#     arr_raw = np.array(arr_raw) # Make sure the array is a numpy array
    
#     # COUNT THE SWITCH NUMBER
#     arr = arr_raw[arr_raw!=0] # Trim zero values
    
#     arr = arr[np.logical_not(np.isnan(arr))] # Trim np.nan
    
#     arr = trim_consecutive_equal(arr) # Trim consecutive equal
    
#     if len(arr) == 0:
#         switch = 0
#     else:
#         switch = int(len(arr)-1)   
    
#     return switch
# #-----------------------------------------------------------------------------#
# #------------------------------------TEST-------------------------------------#
# # arr = np.array([0, 0, 1, 0, 0, -1, -1, 0, 0, 0, 1, 0, 0, 0, 1,1, 1, 0, 0, 0, -1, -1, -1, 1, 1, 1, 0, 1, 0, 0, 0])
# arr = np.array([np.nan, -1, -1, -1, -1, -1, -1, -1])
# switch = switch_counter(arr)

# print(switch)

#%%888888888888888888888888888888888888888888888888888888888888888888888888888#
def switch_distance(arr):
    
    arr = np.array(arr) # Make sure the array is a numpy array
    
    if np.isnan(arr).any():
        switch_counter = np.nan
        distances = np.nan
    else:
        # Find the indices where the sign changes
        sign_changes = np.where(np.diff(np.sign(arr)))[0]

        # Calculate the distances between sign inversions
        distances = np.diff(sign_changes)
        
        if distances.size==0:
            switch_counter = 0
        else:
            switch_counter = len(distances)+1
        
    return  distances, switch_counter
#-----------------------------------------------------------------------------#
#------------------------------------TEST-------------------------------------#
arr = np.array([0, 0, 1, 0, 0, -1, -1, 0, 0, 0, 1, 0, 0, 0, 1,1, 1, 0, 0, 0, -1, -1, -1, 1, 1, 1, 0, 1, 0, 0, 0])
# arr = np.array([np.nan, -1, -1, -1, -1, -1, -1, -1])
# arr = np.array([-1, -1, -1, -1, -1, -1, -1, -1])   
        
distances, switch_counter = switch_distance(arr)

print(distances)
print(switch_counter)
        


#%%888888888888888888888888888888888888888888888888888888888888888888888888888#
# stack_bool = np.array([0,0,0,0,0,1,1,0,0,0,1,0,0,0,1,1,1,0,0,0,1,1,1,1,1,1,0,1,0,0,0]) # Test array
stack_bool = np.array([0, 0, 0, 0, 0, -1, -1, 0, 0, 0, 1, 0, 0, 0, 1,1, 1, 0, 0, 0, -1, -1, -1, 1, 1, 1, 0, 1, 0, 0, 0])  # Test array
# stack_bool = np.array([1, -1,  0,  0,  0,  1,  0,  0,  0]) # Test array
# stack_bool = np.array([0, 0,  1, -1,  0,  0,  0,  -1,  0,  0]) # Test array

stack_bool = np.expand_dims(stack_bool, axis=1)
stack_bool = np.expand_dims(stack_bool, axis=1)

# INITIALIZE STACK AND ARRAY
# activation time stack contains the time between switches. The first layer of this stack contains the first sctivation time that is a lower limit in time because we ignore how long the pixel has keept the same nature in the past.
act_time_stack = np.zeros(stack_bool.shape)
# This is the 2D matrix that collect the number of switch over time
switch_matrix = np.zeros(stack_bool.shape[1:])

consecutive_ones_stack = np.zeros(stack_bool.shape)
consecutive_zeros_stack = np.zeros(stack_bool.shape)
consecutive_minus_ones_stack = np.zeros(stack_bool.shape)

# This is a matrix that collect all the pixel locations that have never been active
never_active_matrix = np.zeros(stack_bool.shape[1:])

dim_t, dim_y, dim_x = stack_bool.shape



for x in range(0, dim_x):
    for y in range(0, dim_y):
        # Slice the stack in a single pixel array where data is collected over time
        slice_array = stack_bool[:, y, x]
        
        analysis_list = ['switch_number', 'consecutive_numbers']
        
        if 'consecutive_numbers' in analysis_list:
            raw_slice_array = stack_bool[:, y, x]
            consecutive_ones_array = find_consecutive_number_lengths(
                raw_slice_array, 1)
            consecutive_zeros_array = find_consecutive_number_lengths(
                raw_slice_array, 0)
            consecutive_minus_ones_array = find_consecutive_number_lengths(
                raw_slice_array, -1)

        # This is the number of zero values before the first non-zero value in the sliced array. It is initialized to zero.
        n_zero = 0
        time_array = []
        '''Check if the sliced array has np.nan. If so, fill this array with
        np.nan and then fill the act_time_stack. This avoids to include in the
        computation pixel that are near the domain edge or pixels that during
        the runs fail in signal detection'''
        if np.isnan(slice_array).any():  # check if a has np.nan value, if so fill matrix with np.nan
            switch_matrix[y, x] = np.nan
            act_time_stack[:, y, x] = np.nan
            pass

        ''' This part treats the cases in which the first entry of the sliced
            array is zero.
        1. If the entire array is full of zeros means that in that position
            nothing occours and the active periods array will be filled with
            np.nan
        2. If the first entry is zero and there are non-zero value in the array
            the script count the number of zeros before the first non-zero
            value and then trim from the slice array the zeros before the first
            non-zero value. Now the resized slice array is ready to go on to
            the next section.
        '''
        if slice_array[0] == 0:
            if np.all(slice_array == 0):  # Check if the sliced array is full of zeros
                # Fill the array with np.nan to keep them transparent
                time_array = np.array([np.nan])
                never_active_matrix[x, y] = 1
            else:
                # Number of zero values before the first non-zero value
                n_zero = np.array(np.where(slice_array != 0))[0, 0]
                # Trim the zero values before the first non-zero value.
                slice_array = slice_array[n_zero:]

        ''' This part treats the cases in which the first entry of the sliced
            array is a non-zero value.
            1. The counter is then initialize to one and the first entry sign
                is detected as the target sign
            2. The script takes all the adjacent elements of the slice array
                and detect the sign of each one.
                a. If the two adjacent elements havethe same sign, the counter
                    is updated with +1
                b. Elif if both the second element is zero and the first or the
                second element have the target sign the counter is +1
                c. Elif a switch occours, so there is a change in sign in the
                    two adjacent elements, or the second element shows a sign
                    that is different from the target sign the counter values
                    is stored in the time array, the target sign change and the
                    count is updated to 1.
            3. The script create the time_array trimming the last period
                because it is a lover boundary as well as the zeros before the
                first non-zero entry.
                
        '''
        if slice_array[0] != 0:  # If the first entry of the sliced array is non-zero
            period_count = 1  # Initialize the count variable. This variable will count the number of activation instants
            switch_count = 0
            # This variable collects the sign of the first element of each same-nature period
            target_sign = np.sign(slice_array[0])
            for i in range(0, len(slice_array)-1):  # Loop over the sliced array
                # a1 and a2 are the two adjacent element in the sliced array
                a1, a2 = slice_array[i], slice_array[i+1]

                if np.sign(a1) == np.sign(a2):  # If two consecutive elements have the same naure
                    period_count += 1  # If two consecutive elements have the same nature the count increases

                elif np.sign(a1)*np.sign(a2) == 0 and (np.sign(a2) == target_sign or np.sign(a1) == target_sign):
                    period_count += 1  # The count increases also if one or both elements are zero but the non-zero value has the target sign

                # The count stops when a switch occours or when one of the two elements shows a sign different from the target sign
                elif np.sign(a1) != np.sign(a2) and (np.sign(a2) != target_sign or np.sign(a1) != target_sign):
                    # This operation append to time_array the count value with his sign. This could be useful to keep trace of the nature of the period.
                    time_array = np.append(
                        time_array, period_count*target_sign)
                    target_sign = -1*target_sign  # Update the target sign
                    period_count = 1  # Update the count variable that will starts again from zero
                    switch_count += 1
                    pass

            # By now the last period is not calculated (actually because, as the first one, it is only a lower boundary of time because it doesn't appear within two switches) so this operation appends this value manually
            time_array = np.append(
                time_array, (len(slice_array)-np.sum(np.abs(time_array)))*target_sign)
            # time_array[0] = time_array[0] + np.sign(time_array[0])*n_zero # Ths operation append, if present, the number of zeroes before the first non-zero value calculated on the very first sliced array (n_zero variable)

        # TRIM THE LAST PERIOD BECAUSE IT IS NOT RELIABLE
        # This number correspond to the index of the last period in the time_array that is not reliable (as the first one)
        ind = np.max(np.where(time_array != 0))
        # So in the filling process I want to exclude the last period:
        # This operation fills the stack with time_array
        act_time_stack[:ind, y, x] = time_array[:ind]

        # SLICE ARRAY WITH NO SWITCH LEADS TO A TIME ARRAY WITH NO PERIODS
        if len(time_array) == 0 or len(time_array) == 1:  # If sliced array does not contain any switch (so if the length of the time array is 0 in the case we do not consider the last period - see above - or if the length is 1 so only one period is considered - the last one - this operation fills the output matrix with np.nan)
            # switch_matrix[y,x] = np.nan # Fill switch matrix with np.nan
            # Fill activation time stack with np.nan
            for t in range(0, act_time_stack.shape[0]):
                act_time_stack[t, y, x] = np.nan

            ''' This section is introducted to eliminate to the dataset pixels that
            show only one switch and so where only the first period is considered'''
        elif len(time_array) == 2:  # If only one switch occours, two periods were identified. This two periods are not completely reliable since they were not observed between two switches
            # switch_matrix[y,x] = np.nan # Fill switch matrix with np.nan
            # Fill activation time stack with np.nan
            for t in range(0, act_time_stack.shape[0]):
                act_time_stack[t, y, x] = np.nan

        else:  # If at least one (or two) switches occours
            # Fill switch matrix
            # switch_matrix[y,x] = len(time_array[time_array!=0]) # To provide the number of switch we decide to trim the zero values to keep only the number of switch

            # Fill activation time stack
            # act_time_stack[:len(time_array),y,x]=time_array # To provide the time between each detected switch
            # This number correspond to the index of the last period in the time_array that is not reliable (as the first one)
            ind = np.max(np.where(time_array != 0))
            # So in the filling process I want to exclude the last period:
            # To provide the time between each detected switch
            act_time_stack[:ind, y, x] = time_array[:ind]

        # FILL consecutive_ones_stack
        consecutive_ones_stack[:len(
            consecutive_ones_array), y, x] = consecutive_ones_array

        # FILL consecutive_zeros_stack
        consecutive_zeros_stack[:len(
            consecutive_zeros_array), y, x] = consecutive_zeros_array

        # FILL consecutive_minus_ones_stack
        consecutive_minus_ones_stack[:len(
            consecutive_minus_ones_array), y, x] = consecutive_minus_ones_array

        # FILL THE SWITCH MATRIX
        switch_matrix[y, x] = switch_count

switch_dist_stack = act_time_stack[1:, :, :]

print('Raw slice', raw_slice_array)
print('Trimmed slice: ', slice_array)
print('Period length:', act_time_stack[:, 0, 0])
print('Distance between switches: ', switch_dist_stack[:, 0, 0])
print('Consecutive 1:  ', consecutive_ones_stack[:, 0, 0])
print('Consecutive 0:  ', consecutive_zeros_stack[:, 0, 0])
print('Consecutive -1: ', consecutive_minus_ones_stack[:, 0, 0])


#%%