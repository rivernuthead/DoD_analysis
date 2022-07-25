#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 11:57:25 2022

@author: erri
"""

import numpy as np

a = np.array([ 1., -1., -1.,  1.,  1.,  1., -1., -1.,  1.])

for t in range(0,len(a)):
    print(t, a[t])

if a[0]!=0:
    x1 = np.array(np.where(a*a[0]==-1)) # indices where pixel nature switch occours respectively to the first element
    diff = x1[:,1:]-x1[:,:-1] # Distance between one element and the consecutive one: split groups
    diff_bool = np.where(diff==1,0,1) # Eliminate consecutive element with the same nature
    p1=np.append(np.array([1]),diff_bool) # Insert 1 for the firs element
    time1_array = x1*p1 # Apply filter
    
    
    x2 = np.array(np.where(a[1:]*a[0]==1))+1
    diff2 = x2[:,1:]-x2[:,:-1]
    diff2_bool = np.where(diff2==1,0,1)
    p2=np.append(np.array([1]),diff2_bool)
    time2_array = x2*p2
    
    
    time_array = np.sort(np.append(time1_array, time2_array))
    time_array = time_array[time_array != 0]
    if a[time_array[0]]*a[0]==1:
        print('True')
        time_array = time_array[1:]
    
    
elif a[0]==0:
    count=1
    for i in range(1,len(a)):
        if a[i]==0:
            count+=1
        else:
            break
    a=a[count:]
    x1 = np.array(np.where(a*a[0]==-1)) # indices where pixel nature switch occours respectively to the first element
    diff = x1[:,1:]-x1[:,:-1] # Distance between one element and the consecutive one: split groups
    diff_bool = np.where(diff==1,0,1) # Eliminate consecutive element with the same nature
    p1=np.append(np.array([1]),diff_bool) # Insert 1 for the firs element
    time1_array = x1*p1 # Apply filter
    
    
    x2 = np.array(np.where(a[1:]*a[0]==1))+1
    diff2 = x2[:,1:]-x2[:,:-1]
    diff2_bool = np.where(diff2==1,0,1)
    p2=np.append(np.array([1]),diff2_bool)
    time2_array = x2*p2
    
    time_array = np.sort(np.append(time1_array, time2_array))
    time_array = time_array[time_array != 0]
    if a[time_array[0]]*a[0]==1:
        print('True')
        time_array = time_array[1:]
    time_array = time_array + count # Trim zero values
    
