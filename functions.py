# IMPORT LIBRARIES
import os
import numpy as np
import scipy.ndimage
import cv2



def downsample_matrix_interpolation(matrix, factor):
    '''
    INPUT:
        matrix: numpy array
        factor: int, is the downscaling factor

    OUTPUT:
        downsampled_matrix
    Description:
        This funcrtion compute the downsacling of a matrix
    '''
    
    # Get the shape of the original matrix

    height, width = matrix.shape

    # Calculate the new dimensions after downsampling
    new_height = height // factor
    new_width = width // factor

    # Create a grid of coordinates for the new downsampling points
    row_indices = np.linspace(0, height - 1, new_height)
    col_indices = np.linspace(0, width - 1, new_width)

    # Perform bilinear interpolation to estimate the values at new points
    downsampled_matrix = scipy.ndimage.ndimage.map_coordinates(matrix, 
                                                 np.meshgrid(row_indices, col_indices),
                                                 order=1,
                                                 mode='nearest')

    # Reshape the downsampled matrix to the new dimensions
    downsampled_matrix = downsampled_matrix.reshape(new_height, new_width)

    return downsampled_matrix


def img_scaling_to_DEM(image, scale, dx, dy, rot_angle):
    '''
    INPUT:
        image: numpy array
        scale: real, is the scaling factor
        dx: int, traslation along the horizontal axes
        dy: int, traslation along the vertical axes
        rot_angle: real, degree
    OUTPUT:
        downsampled_matrix
    Description:
        This funcrtion perform the roto-scale-traslation of an image given as a numpy array
    '''

    # Create the transformation matrix
    M = np.float32([[scale, 0, dx], [0, scale, dy]])
    
    # Apply the transformation to img1 and store the result in img2
    rows, cols = image.shape
    image_rsh = cv2.warpAffine(image, M, (cols, rows))
    
    # Rotate the image
    
    M = cv2.getRotationMatrix2D((image_rsh.shape[1]/2, image_rsh.shape[0]/2), rot_angle, 1)
    image_rsh = cv2.warpAffine(image_rsh, M, (image_rsh.shape[1], image_rsh.shape[0]))
    
    # Trim zeros rows and columns due to shifting
    x_lim, y_lim = dx+int(cols*scale), dy+int(rows*scale)
    image_rsh = image_rsh[:y_lim, :x_lim]

    return image_rsh


def downsample_matrix(matrix, kernel_size, factor):
    # Perform convolution with the kernel for block averaging
    kernel = np.ones((kernel_size, kernel_size)) / (kernel_size ** 2)
    convolved_matrix = convolve2d(matrix, kernel, mode='same', boundary='symm')

    # Perform downsampling with the given factor
    downsampled_matrix = convolved_matrix[::factor, ::factor]

    return downsampled_matrix

def check_integer(number):
    if isinstance(number, int):
        print(f"{number} is an integer.")
    else:
        print(f"{number} is not an integer.")



def non_overlapping_average(image, kernel_size):
    # Get the shape of the input image
    height, width = image.shape

    if isinstance(height / kernel_size, int) or isinstance(width / kernel_size, int):
        print("Warning: kernel size does not fit the image size: the function will ignore the remaining pixels at the right and bottom edges")
    # else:
    #     print(f"{number} is not an integer.")

    # Calculate the new dimensions for the non-overlapping blocks
    new_height = height // kernel_size
    new_width = width // kernel_size

    # Reshape the image into non-overlapping blocks
    blocks = image[:new_height * kernel_size, :new_width * kernel_size].reshape(new_height, kernel_size, new_width, kernel_size)

    # Calculate the average within each block
    block_averages = blocks.mean(axis=(1, 3))

    return block_averages


def update_matrix_dimensions(matrix1, matrix2):
    # Calculate the maximum dimensions
    max_rows = max(matrix1.shape[0], matrix2.shape[0])
    max_cols = max(matrix1.shape[1], matrix2.shape[1])

    # Create new matrices with NaN values and the new dimensions
    updated_matrix1 = np.full((max_rows, max_cols), np.nan)
    updated_matrix2 = np.full((max_rows, max_cols), np.nan)

    # Copy the original data to the new matrices
    updated_matrix1[:matrix1.shape[0], :matrix1.shape[1]] = matrix1
    updated_matrix2[:matrix2.shape[0], :matrix2.shape[1]] = matrix2

    return updated_matrix1, updated_matrix2

def cut_matrices_to_minimum_dimension(matrix1, matrix2):
    # Calculate the minimum dimensions (number of columns and rows)
    min_cols = min(matrix1.shape[1], matrix2.shape[1])
    min_rows = min(matrix1.shape[0], matrix2.shape[0])
    
    # Cut both matrices to match the new dimensions
    matrix1_cut = matrix1[:min_rows, :min_cols]
    matrix2_cut = matrix2[:min_rows, :min_cols]
    
    return matrix1_cut, matrix2_cut