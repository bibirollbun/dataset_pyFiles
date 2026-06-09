import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
is_train=True


if is_train:
    # Extract and load 3D point coordinates and colors
    points = np.load("/kaggle/input/stairs-point-coud/point1.npy")
    color = np.load("/kaggle/input/stairs-point-coud/color1.npy")

    data=points
    
    # Separate the coordinates
    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]
    r = color[:, 0]
    g = color[:, 1]
    b = color[:, 2]
    
    # Normalize RGB values to the range [0, 1]
    colors = ['rgb({},{},{})'.format(r[i], g[i], b[i]) for i in range(len(r))]
    
    # Create the 3D scatter plot
    fig = go.Figure(data=[go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(
            size=1,
            color=colors,
        )
    )])
    
    # Set plot layout
    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        title='3D Point Cloud with Colors'
    )
    
    # Show the plot
    fig.show()

