#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 12:12:39 2023

@author: erri
"""
import numpy as np

'''
DoD input stack structure:
    
    DoD_stack[time,y,x,delta]
    DoD_stack_bool[time,y,x,delta]
    
     - - - 0 - - - - 1 - - - - 2 - - - - 3 - - - - 4 - - - - 5 - - - - 6 - - - - 7 - - - - 8 - -  >    delta
  0  |  DoD 1-0   DoD 2-0   DoD 3-0   DoD 4-0   DoD 5-0   DoD 6-0   DoD 7-0   DoD 8-0   DoD 9-0
  1  |  DoD 2-1   DoD 3-1   DoD 4-1   DoD 5-1   DoD 6-1   DoD 7-1   DoD 8-1   DoD 9-1
  2  |  DoD 3-2   DoD 4-2   DoD 5-2   DoD 6-2   DoD 7-2   DoD 8-2   DoD 9-2
  3  |  DoD 4-3   DoD 5-3   DoD 6-3   DoD 7-3   DoD 8-3   DoD 9-3
  4  |  DoD 5-4   DoD 6-4   DoD 7-4   DoD 8-4   DoD 9-4
  5  |  DoD 6-5   DoD 7-5   DoD 8-5   DoD 9-5
  6  |  DoD 7-6   DoD 8-6   DoD 9-6
  7  |  DoD 8-7   DoD 9-7
  8  |  DoD 9-8
     |
     v
    
     time
        
'''

for delta in range(0,8):
    array = np.linspace(0,8, 9)
    if delta!=0:
        array = array[:-delta]
    else:
        pass
    print(array)
    print(delta+1)
    
    print(array//(delta+1))
    # print(delta+2)
    
    print(np.max(array//(delta+1))*(delta+1))

    # print(np.max((array//(delta+2))*(delta+2)))

    
    # test = np.max(array//(delta+1))*(delta+1)
    
    # print(test)
    
    print()