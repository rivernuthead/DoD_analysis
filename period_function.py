#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 15:47:48 2023

@author: erri
"""

import numpy as np

# stack_bool = np.array([0,0,0,0,0,1,1,0,0,0,1,0,0,0,1,1,1,0,0,0,1,1,1,1,1,1,0,1,0,0,0]) # Test array
# stack_bool = np.array([0,0,0,0,0,-1,-1,0,0,0,1,0,0,0,1,1,1,0,0,0,-1,-1,-1,1,1,1,0,1,0,0,0]) # Test array
stack_bool = np.array([1, -1,  0,  0,  0,  1,  0,  0,  0]) # Test array
stack_bool = np.array([0, 0,  1, -1,  0,  0,  0,  -1,  0,  0]) # Test array

stack_bool = np.expand_dims(stack_bool, axis=1)
stack_bool = np.expand_dims(stack_bool, axis=1)

# INITIALIZE STACK AND ARRAY
act_time_stack = np.zeros(stack_bool.shape) # activation time stack contains the time between switches. The first layer of this stack contains the first sctivation time that is a lower limit in time because we ignore how long the pixel has keept the same nature in the past.
switch_matrix = np.zeros(stack_bool.shape[1:]) # This is the 2D matrix that collect the number of switch over time


dim_t, dim_y, dim_x = stack_bool.shape

mode = 'switch_number'

for x in range(0,dim_x):
    for y in range(0,dim_y):
        slice_array = stack_bool[:,y,x] # Slice the stack in a single pixel array where data is collected over time
        n_zero = 0 # This is the number of zero values before the first non-zero value in the sliced array. It is initialized to zero.
        time_array = []
        '''Check if the sliced array has np.nan. If so, fill this array with
        np.nan and then fill the act_time_stack. This avoids to include in the
        computation pixel that are near the domain edge or pixels that during
        the runs fail in signal detection'''
        if np.isnan(slice_array).any(): # check if a has np.nan value, if so fill matrix with np.nan
            switch_matrix[y,x] = np.nan
            act_time_stack[:,y,x] = np.nan
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
        if slice_array[0]==0:
            if np.all(slice_array == 0): # Check if the sliced array is full of zeros
                time_array = np.array([np.nan]) # Fill the array with np.nan to keep them transparent
            else:
                n_zero=np.array(np.where(slice_array!=0))[0,0] # Number of zero values before the first non-zero value
                slice_array=slice_array[n_zero:] # Trim the zero values before the first non-zero value.
        
                
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
        if slice_array[0]!=0: # If the first entry of the sliced array is non-zero
            period_count=1 # Initialize the count variable. This variable will count the number of activation instants
            switch_count=0
            target_sign = np.sign(slice_array[0]) # This variable collects the sign of the first element of each same-nature period
            for i in range(0,len(slice_array)-1): # Loop over the sliced array
                a1, a2 = slice_array[i], slice_array[i+1] # a1 and a2 are the two adjacent element in the sliced array
                
                if np.sign(a1)==np.sign(a2): # If two consecutive elements have the same naure
                    period_count += 1 # If two consecutive elements have the same nature the count increases
                
                elif np.sign(a1)*np.sign(a2)==0 and (np.sign(a2)==target_sign or np.sign(a1)==target_sign):
                    period_count += 1 # The count increases also if one or both elements are zero but the non-zero value has the target sign
                
                elif np.sign(a1)!=np.sign(a2) and (np.sign(a2)!=target_sign or np.sign(a1)!=target_sign): # The count stops when a switch occours or when one of the two elements shows a sign different from the target sign
                    time_array = np.append(time_array, period_count*target_sign)  # This operation append to time_array the count value with his sign. This could be useful to keep trace of the nature of the period.
                    target_sign=-1*target_sign # Update the target sign
                    period_count=1 # Update the count variable that will starts again from zero
                    switch_count+=1
                    pass
                    
            time_array = np.append(time_array, (len(slice_array)-np.sum(np.abs(time_array)))*target_sign) # By now the last period is not calculated (actually because, as the first one, it is only a lower boundary of time because it doesn't appear within two switches) so this operation appends this value manually
            # time_array[0] = time_array[0] + np.sign(time_array[0])*n_zero # Ths operation append, if present, the number of zeroes before the first non-zero value calculated on the very first sliced array (n_zero variable)
        
        # TRIM THE LAST PERIOD BECAUSE IT IS NOT RELIABLE
        ind = np.max(np.where(time_array!=0)) # This number correspond to the index of the last period in the time_array that is not reliable (as the first one)
        # So in the filling process I want to exclude the last period:
        act_time_stack[:ind,y,x]=time_array[:ind] # This operation fills the stack with time_array     
        
        # SLICE ARRAY WITH NO SWITCH LEADS TO A TIME ARRAY WITH NO PERIODS
        if len(time_array)==0 or len(time_array)==1: # If sliced array does not contain any switch (so if the length of the time array is 0 in the case we do not consider the last period - see above - or if the length is 1 so only one period is considered - the last one - this operation fills the output matrix with np.nan)
            # switch_matrix[y,x] = np.nan # Fill switch matrix with np.nan
            for t in range(0,act_time_stack.shape[0]): # Fill activation time stack with np.nan
                act_time_stack[t,y,x]=np.nan
        
            ''' This section is introducted to eliminate to the dataset pixels that
            show only one switch and so where only the first period is considered'''
        elif len(time_array)==2: # If only one switch occours, two periods were identified. This two periods are not completely reliable since they were not observed between two switches
            switch_matrix[y,x] = np.nan # Fill switch matrix with np.nan
            for t in range(0,act_time_stack.shape[0]): # Fill activation time stack with np.nan
                act_time_stack[t,y,x]=np.nan
            
        else: # If at least one (or two) switches occours
            # Fill switch matrix
            switch_matrix[y,x] = len(time_array[time_array!=0]) # To provide the number of switch we decide to trim the zero values to keep only the number of switch
            
            # Fill activation time stack
            # act_time_stack[:len(time_array),y,x]=time_array # To provide the time between each detected switch
            ind = np.max(np.where(time_array!=0)) # This number correspond to the index of the last period in the time_array that is not reliable (as the first one)
            # So in the filling process I want to exclude the last period:
            act_time_stack[:ind,y,x]=time_array[:ind] # To provide the time between each detected switch
print(slice_array)        
print(act_time_stack[:,0,0])