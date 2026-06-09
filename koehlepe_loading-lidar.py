!pip install laspy lazrs geopandas -q


import laspy
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from IPython.display import Image, display

def read_las_to_numpy(las_file_path):
    las = laspy.read(las_file_path)
    points_struct = las.points
    try:
        ground_mask = (points_struct['classification'] == 2)
    except ValueError:
        print("Classification field not found or named differently.")
        ground_mask = np.ones(len(points_struct), dtype=bool)
    
    ground_points_struct = points_struct[ground_mask]
    
    # Extract the X, Y, and Z coordinates from the ground points
    try:
        ground_points_xyz = np.vstack((ground_points_struct['x'], ground_points_struct['y'], ground_points_struct['z'])).transpose()
    except ValueError:
        ground_points_xyz = np.vstack((ground_points_struct['X'], ground_points_struct['Y'], ground_points_struct['Z'])).transpose()
    return ground_points_xyz

def get_interpolated_data(raw):
    points = raw[:, :2]  # X and Y coordinates as interpolation points
    values = raw[:, 2]   # Z values as values to interpolate

    # --- Define the interpolation grid ---
    # Determine the grid extent based on the ground points
    x_min, y_min = np.min(points, axis=0)
    x_max, y_max = np.max(points, axis=0)

    # Define the grid resolution (adjust as needed)
    grid_resolution = 1.0  # e.g., 1 meter resolution

    # Create the grid coordinates
    grid_x, grid_y = np.mgrid[x_min:x_max:grid_resolution, y_min:y_max:grid_resolution]
    try:
        interpolated_grid = griddata(points, values, (grid_x, grid_y), method='linear')
    except ValueError:
        print("Linear interpolation failed, trying 'nearest' method.")
        interpolated_grid = griddata(points, values, (grid_x, grid_y), method='nearest')
    return interpolated_grid

def plot_grid(grid):
    plt.figure(figsize=(10, 8))
    plt.imshow(interpolated_grid.T,origin='lower', cmap='viridis')
    plt.colorbar(label='Interpolated Elevation (Z)')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Interpolated Ground Surface (DEM)')
    plt.grid(False)
    plt.show()


import os
import glob

laz_files_path = "/kaggle/input/lidar-survey-on-1173-hectares-in-jamari-2013/JAM_A03_2013_LiDAR.zip/JAM_A03_2013_LiDAR/JAM_A03_2013_LiDAR/JAM_A03_2013_laz/"
laz_files = glob.glob(os.path.join(laz_files_path, "*.laz"))
count = 0
for laz_file in laz_files:
    if count==0:
        ground_points = read_las_to_numpy(laz_file)
        count = 1
    else:
        ground_points = np.vstack((ground_points,read_las_to_numpy(laz_file)))



indices = np.random.choice(ground_points.shape[0], size=1000000,replace=False)
samples = ground_points[indices,:]
interpolated_grid = get_interpolated_data(samples)
plot_grid(interpolated_grid)

