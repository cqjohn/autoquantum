import numpy as np
from math import sqrt

width = 600
height = 400

#generate a 2D NumPy array of shape (width, height) filled with random values sampled from a uniform distribution over [0, 1) (excluding 1).
input_frame = np.random.rand(width, height)


def locate_first_point(arr, threshold, width, height): #does this function find the first transition line or triple point?
    """
    This function locates the first point in a 2D array (arr) that has a value greater than a specified threshold.
    :param arr: The input 2D array.
    :param threshold: The specified threshold value.
    :param width: Width of the array.
    :param height: Height of the array.
    :return: A tuple containing the coordinates of the point with the minimum distance from the origin, the actual distance, and the value at that point in the array.
    """
    # List to store coordinates of points with values greater than the threshold. How do we set this threshold?
    strong_signal_coords = []

    #iterate over each value in arr and if the value at a given position is greater than the specified threshold, its coordinates [i, j] are added to the strong_signal_coords list
    for i in range(width):
        for j in range(height):
            if arr[i][j] > threshold:
                strong_signal_coords.append([i, j])

    # Print the number of points with values greater than the threshold.
    print(len(strong_signal_coords))

    # Initialize variables for minimum distance and corresponding coordinates.
    min_dist_from_origin = height + width   #this value ends up getting updated  #why height+width?  i think because this value is getting updated it does not matter
    min_coords = []

    # Iterate over strong_signal_coords to find the minimum distance from the origin.
    for coords in strong_signal_coords:
        # print(coords)
        x = coords[0]
        y = coords[1]
        # calculate distance from lower left corner: (0, height)
        dist_from_origin = sqrt((x)**2 + (y)**2)
        # print(dist_from_origin)
        # input()

        # Update minimum distance and corresponding coordinates if a shorter distance is found.
        if dist_from_origin < min_dist_from_origin:
            min_dist_from_origin = dist_from_origin
            min_coords = coords
    # print(min_coords, min_dist_from_origin, arr[min_coords[0]][min_coords[1]])

    # Return the coordinates, distance, and value at the minimum distance point.
    return min_coords, min_dist_from_origin,arr[min_coords[0]][min_coords[1]]


def locate_nearby_max_point(arr, radius, width, height, x, y):
    """
    This function locates the point with the maximum value in a specified neighborhood around the given coordinates.
    :param arr: The input 2D array.
    :param radius: The specified radius defining the neighborhood.
    :param width: Width of the array.
    :param height: Height of the array.
    :param x: X-coordinate of the specified point.
    :param y: Y-coordinate of the specified point.
    :return: A tuple containing the coordinates of the point with the maximum value in the specified neighborhood and the actual maximum value.
    """

    # Initialize variables for maximum value and corresponding coordinates.
    max_value = arr[x][y]
    max_coords = [x,y]

    # Iterate over a neighborhood defined by radius to find the maximum value and corresponding coordinates.
    for i in range(radius):
        for j in range(radius):

            # Check boundaries to avoid index out of range errors.
            if x-i < 0 or y-j < 0 or x+i > radius - 1 or y+j > radius - 1:
                continue

            # Update maximum value and coordinates if a higher value is found within the neighborhood.
            if arr[x-i][y-j] > max_value:
                max_value = arr[x-i][y-j]
                max_coords = [x-i, y-j]
            if arr[x+i][y+j] > max_value:
                max_value = arr[x+i][y+j]
                max_coords = [x+i, y+j]

    # Return the coordinates and the maximum value in the specified neighborhood.
    return max_coords, max_value

            

