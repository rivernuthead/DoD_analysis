#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 15:47:48 2023

@author: erri
"""
import numpy as np

#%%
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
import numpy as np

def activity_cluster(arr):
    '''
    This function takes a boolean (-1,0,+1) numpy array as input and convert
    zero values to the sign of the previous cell.

    Parameters
    ----------
    arr : TYPE
        DESCRIPTION.

    Returns
    -------
    out_arr : TYPE
        DESCRIPTION.

    '''
    
    out_arr = np.zeros((arr.shape))
    
    i = 0 # Counter
    first_zeros = 0
    if np.all(arr==0):
        pass
    else:
        while arr[i]==0: # Skip the first set of zeros if present
            i+=1
            first_zeros += 1
        
        for i in range(first_zeros,len(arr)):
            # print(i)
            # print(arr[i])
            if arr[i]!=0:
                sing = np.sign(arr[i])
                out_arr[i] = arr[i]
                pass
            elif arr[i]==0:
                out_arr[i] = sing
            
    return out_arr
#-----------------------------------------------------------------------------#
#------------------------------------TEST-------------------------------------#
# arr = np.array([0, 0, 0, 0, 0, -1, -1, 0, 0, 0, 1, 0, 0, 0, 1,1, 1, 0, 0, 0, -1, -1, -1, 1, 1, 1, 0, 1, 0, 0, 0])
# out_arr = activity_cluster(arr)

# print(arr)
# print(out_arr)

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
    
    if arr[0]==0:
        switch_counter += -1
    return  np.array(distances), switch_counter
#-----------------------------------------------------------------------------#
#------------------------------------TEST-------------------------------------#
# arr = np.array([0, 0, 1, 0, 0, -1, -1, 0, 0, 0, 1, 0, 0, 0, 1,1, 1, 0, 0, 0, -1, -1, -1, 1, 1, 1, 0, 1, 0, 0, 0])
# # arr = np.array([np.nan, -1, -1, -1, -1, -1, -1, -1])
# # arr = np.array([-1, -1, -1, -1, -1, -1, -1, -1])   
        
# distances, switch_counter = switch_distance(arr)

# print(distances)
# print(switch_counter)
        
#%%888888888888888888888888888888888888888888888888888888888888888888888888888#
import numpy as np

def sign_inversion_distances(arr):
    # Find the indices where the sign changes
    sign_changes = np.where(np.diff(np.sign(arr)))[0]

    # Calculate the distances between sign inversions
    distances = np.diff(sign_changes)
    
    switch_counter = len(distances)+1
    return distances, switch_counter

#-----------------------------------------------------------------------------#
#------------------------------------TEST-------------------------------------#
# input_array = np.array([1, -2, 3, 0, 5, 6, -7, 8, -9, 10])
# distances, switch_counter = sign_inversion_distances(input_array)

# print("Array:", input_array)
# print("Distances between sign inversions:", distances)
# print("Number of switch", switch_counter)

#%%888888888888888888888888888888888888888888888888888888888888888888888888888#

import numpy as np

# stack_bool = np.array([0,0,0,0,0,1,1,0,0,0,1,0,0,0,1,1,1,0,0,0,1,1,1,1,1,1,0,1,0,0,0]) # Test array
raw_slice_array = np.array([0, 0, 0, 0, 0, -1, -1, 0, 0, 0, 1, 0, 0, 0, 1,1, 1, 0, 0, 0, -1, -1, -1, 1, 1, 1, 0, 1, 0, 0, 0])  # Test array
# stack_bool = np.array([1, -1,  0,  0,  0,  1,  0,  0,  0]) # Test array
# stack_bool = np.array([0, 0,  1, -1,  0,  0,  0,  -1,  0,  0]) # Test array

def switch_period_analysis(arr, analysis_list):
    raw_slice_array = arr
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
    if np.isnan(raw_slice_array).any():  # check if a has np.nan value, if so fill matrix with np.nan
        time_array = np.array([np.nan])
        pass

    
    if raw_slice_array[0] == 0:
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
        if np.all(raw_slice_array == 0):  # Check if the sliced array is full of zeros
            # Fill the array with np.nan to keep them transparent
            time_array = np.array([np.nan])
        else:
            # Number of zero values before the first non-zero value
            n_zero = np.array(np.where(raw_slice_array != 0))[0, 0]
            # Trim the zero values before the first non-zero value.
            raw_slice_array = raw_slice_array[n_zero:]

    
    if raw_slice_array[0] != 0:
        ''' This part treats all the sliced array. If the raw array starts with
        zero the above section has count the number of zeros and has trimmed
        them out. So, at this point all the sliced arrays are starting with
        a nn-zero value.
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
        period_count = 1  # Initialize the count variable. This variable will count the number of activation instants
        switch_count = 0
        # This variable collects the sign of the first element of each same-nature period
        target_sign = np.sign(raw_slice_array[0])
        for i in range(0, len(raw_slice_array)-1):  # Loop over the sliced array
            # a1 and a2 are the two adjacent element in the sliced array
            a1, a2 = raw_slice_array[i], raw_slice_array[i+1]

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
        time_array = np.append(time_array, (len(raw_slice_array)-np.sum(np.abs(time_array)))*target_sign)
        # time_array[0] = time_array[0] + np.sign(time_array[0])*n_zero # Ths operation append, if present, the number of zeroes before the first non-zero value calculated on the very first sliced array (n_zero variable)
    
    return np.array(time_array), np.array(consecutive_ones_array), np.array(consecutive_zeros_array), np.array(consecutive_minus_ones_array), n_zero

#-----------------------------------------------------------------------------#
#------------------------------------TEST-------------------------------------#
          
analysis_list = ['switch_number', 'consecutive_numbers']

time_array, consecutive_ones_array, consecutive_zeros_array, consecutive_minus_ones_array, n_zero = switch_period_analysis(raw_slice_array, analysis_list)

distances_array, switch_counter = switch_distance(activity_cluster(raw_slice_array))
            


print(raw_slice_array)
print(n_zero)
print(time_array)
print(consecutive_minus_ones_array)
print(consecutive_zeros_array)
print(consecutive_ones_array)
print(distances_array)
print(switch_counter)

'''
n_0: int - Number of zeros before the first non-zero value
time_array: np.array - Length with sign of the active periods
consecutive_minus_ones_array: np.array - Length of consecutive -1 values
consecutive_zeros_array: np.array - Length of consecutive 0 values
consecutive_ones_array: np.array - Length of consecutive +1 values
distances_array: np.array - Distance between switches
switch_counter: int - Number of switches
'''
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