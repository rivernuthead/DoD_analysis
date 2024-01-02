def img_scaling_to_DEM(image, scale, dx, dy, rot_angle):

    # Create the transformation matrix
    M = np.float32([[scale, 0, dx], [0, scale, dy]])

    # Apply the transformation to img1 and store the result in img2
    rows, cols = image.shape
    image_rsh = cv2.warpAffine(image, M, (cols, rows))

    # Rotate the image

    M = cv2.getRotationMatrix2D(
        (image_rsh.shape[1]/2, image_rsh.shape[0]/2), rot_angle, 1)
    image_rsh = cv2.warpAffine(
        image_rsh, M, (image_rsh.shape[1], image_rsh.shape[0]))

    # Trim zeros rows and columns due to shifting
    x_lim, y_lim = dx+int(cols*scale), dy+int(rows*scale)
    image_rsh = image_rsh[:y_lim, :x_lim]

    return image_rsh


def downsample_matrix_interpolation(matrix, factor):
    # Get the shape of the original matrix
    height, width = matrix.shape

    # Calculate the new dimensions after downsampling
    new_height = height // factor
    new_width = width // factor

    # Create a grid of coordinates for the new downsampling points
    row_indices = np.linspace(0, height - 1, new_height)
    col_indices = np.linspace(0, width - 1, new_width)

    # Perform bilinear interpolation to estimate the values at new points
    downsampled_matrix = ndimage.map_coordinates(matrix,
                                                 np.meshgrid(
                                                     row_indices, col_indices),
                                                 order=1,
                                                 mode='nearest')

    # Reshape the downsampled matrix to the new dimensions
    downsampled_matrix = downsampled_matrix.reshape(new_height, new_width)

    return downsampled_matrix
