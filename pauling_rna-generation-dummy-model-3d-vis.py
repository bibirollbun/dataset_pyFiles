### Loading Relevant Packages

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
from mpl_toolkits.mplot3d import Axes3D


def initial_RNA_input():
    n_vectors = 10
    angle_deg = 140
    length = 6
    return n_vectors, angle_deg, length

n_vectors, angle_deg, length = initial_RNA_input()


# Helper function to generate a new vector at a fixed angle to the previous one
# This new vector is not influenced by the center of all the vectors

def generate_next_vector(v_prev, angle_deg, length_vector = length):
    # Conver v_prev to a unit vector
    v_prev = v_prev / np.linalg.norm(v_prev)  

    # Step 1: Create a random vector
    rand_vec = np.random.randn(3)

    # Step 2: Make it orthogonal to v_prev
    rand_vec -= np.dot(rand_vec, v_prev) * v_prev
    rand_vec /= np.linalg.norm(rand_vec)

    # Step 3: Rotate v_prev toward rand_vec by angle_rad
    # The 180 - angle_deg is to correctly calculate the angle based on how the vectors are pointing
    v_new = np.cos(np.deg2rad(180 - angle_deg)) * v_prev + np.sin(np.deg2rad(180 - angle_deg)) * rand_vec

    # Step 4: Ensure the length of the new vector is correct
    v_new_final = length_vector * (v_new / np.linalg.norm(v_new)) 

    # Step 4: Normalize and scale to desired length
    return v_new_final

random_vector = generate_next_vector([6, 0, 0], angle_deg, length_vector = length)
random_vector


# Initialization

def initialization():
    vectors = []
    v0 = np.array([length, 0, 0])
    vectors.append(v0)
    return vectors

vectors_initialized = initialization() 
vectors_initialized


# Generate the vectors between the C-1 atoms

def generate_vectors(n_vectors, vectors, angle_deg):
    for _ in range(n_vectors - 1):
        new_v = generate_next_vector(vectors[-1], angle_deg)
        vectors.append(new_v)
    return vectors 

vectors = generate_vectors(n_vectors, vectors_initialized, angle_deg)
vectors


### Compute the cumulative positions for the generated vectors of C-1 atoms

def consecutive_positions(vectors):
    positions = [np.zeros(3)]
    for v in vectors:
        positions.append(positions[-1] + v)
    positions = np.array(positions)
    return positions

positions = consecutive_positions(vectors)
positions


def plot3D(positions):
    # Plotting
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], marker='o')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Chain of 3D Vectors with 140° Between Each')
    
    return plt.show()

plot3D(positions)
    


### Check lengths and angles

def angle_between(v1, v2):
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    cos_theta = np.clip(cos_theta, -1.0, 1.0)  # numerical safety
    return np.rad2deg(np.arccos(cos_theta))

def check_angles(vectors):
    # Compute and print all angles
    for i in range(len(vectors) - 1):
        angle = angle_between(vectors[i], -1 * vectors[i + 1])
        print(f"Angle between vector {i} and {i+1}: {angle:.2f} degrees")

def check_lengths(vectors):
    for i, v in enumerate(vectors):
        length = np.linalg.norm(v)
        print(f"Vector {i} length: {length:.6f} {'✅' if np.isclose(length, 6.0, atol=1e-6) else '❌'}")


check_angles(vectors)
check_lengths(vectors)


## Recap of the entire code

### Initial RNA input
n_vectors, angle_deg, length = initial_RNA_input()

### Initialize
vectors_initialized = initialization()

# Generate the new vectors
vectors = generate_vectors(n_vectors, vectors_initialized, angle_deg)

# Compute the cumulative positions
positions = consecutive_positions(vectors)

# Plotting
plot3D(positions)

# Check angles and lengths
check_angles(vectors)
check_lengths(vectors)





