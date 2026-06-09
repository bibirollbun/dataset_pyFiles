!pip install pyvista


import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt
import os


os.chdir("..")


# An example for some temperature field as a function of spatial coordinates
def temp_field(x, y):
    return x**2+y**2


def generate_vtk(solution_func, filename, nx=100, ny=100):
    # Create mesh grid
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)  # For 2D data, Z is set to zero
    
    # Calculate solution
    temperature = solution_func(X, Y)
    
    # Create a PyVista grid
    grid = pv.StructuredGrid(X, Y, Z)
    grid["Temperature"] = temperature.ravel(order="F")  # Flatten for VTK format
    
    # Save the grid to a .vtk file
    grid.save("./working/" + filename + ".vtk")


generate_vtk(temp_field, "Field1")


def load_and_plot_vtk(filename):
    mesh = pv.read("./working/" + filename)
    # Check if the file has the expected data
    print("Available Scalars:", mesh.array_names)
    
    # Plot the temperature field
    if "Temperature" in mesh.array_names:
        points = mesh.points  # (x, y, z) coordinates
        temperature = mesh['Temperature']
        print(points.shape)
        x = points[:, 0].reshape(100, 100)
        y = points[:, 1].reshape(100, 100)
        temperature = temperature.reshape(100, 100)
        plt.contourf(x, y, temperature, cmap='jet', levels=100)
        plt.colorbar()
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Temperature')
        plt.show()
    else:
        print("No 'Temperature' data found in the file.")


load_and_plot_vtk('Field1.vtk')

