# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')


# Read the CSV file
base_path = '/kaggle/input/byu-locating-bacterial-flagellar-motors-2025'
train_dir = '/kaggle/input/byu-locating-bacterial-flagellar-motors-2025/train'

df = pd.read_csv(f'{base_path}/train_labels.csv')

# Display the first few rows of the dataframe
print("\nFirst few rows of the dataset:")
display(df.head())


# Summary statistics
df.info()


# Visualize the distribution of motor locations
plt.figure(figsize=(15, 6))

# Motor axis 0
plt.subplot(1, 3, 1)
motor_axis_0 = df[df['Motor axis 0'] != -1]['Motor axis 0']
sns.histplot(motor_axis_0, kde=True)
plt.title('Distribution of Motor axis 0\n(Excluding no-motor positions)')

# Motor axis 1
plt.subplot(1, 3, 2)
motor_axis_1 = df[df['Motor axis 1'] != -1]['Motor axis 1']
sns.histplot(motor_axis_1, kde=True)
plt.title('Distribution of Motor axis 1\n(Excluding no-motor positions)')

# Motor axis 2
plt.subplot(1, 3, 3)
motor_axis_2 = df[df['Motor axis 2'] != -1]['Motor axis 2']
sns.histplot(motor_axis_2, kde=True)
plt.title('Distribution of Motor axis 2\n(Excluding no-motor positions)')

plt.tight_layout()
plt.show()


# Visualize the scatterplot of motor locations in the xy, xz, and yz axes
plt.figure(figsize=(15, 6))

# XY plane
plt.subplot(1, 3, 1)
sns.scatterplot(x='Motor axis 1', y='Motor axis 2', data=df[df['Motor axis 0'] != -1], alpha=0.5)
plt.title('Motor Locations in XY Plane\n(Excluding no-motor positions)')
plt.xlabel('Motor axis 1')
plt.ylabel('Motor axis 2')

# XZ plane
plt.subplot(1, 3, 2)
sns.scatterplot(x='Motor axis 1', y='Motor axis 0', data=df[df['Motor axis 0'] != -1], alpha=0.5)
plt.title('Motor Locations in XZ Plane\n(Excluding no-motor positions)')
plt.xlabel('Motor axis 1')
plt.ylabel('Motor axis 0')

# YZ plane
plt.subplot(1, 3, 3)
sns.scatterplot(x='Motor axis 2', y='Motor axis 0', data=df[df['Motor axis 0'] != -1], alpha=0.5)
plt.title('Motor Locations in YZ Plane\n(Excluding no-motor positions)')
plt.xlabel('Motor axis 2')
plt.ylabel('Motor axis 0')

plt.tight_layout()
plt.show()


# Visualize the distribution of array shapes
plt.figure(figsize=(15, 6))

# Array shape axis 0
plt.subplot(1, 3, 1)
sns.histplot(df['Array shape (axis 0)'], kde=True)
plt.title('Distribution of Array shape axis 0')

# Array shape axis 1
plt.subplot(1, 3, 2)
sns.histplot(df['Array shape (axis 1)'], kde=True)
plt.title('Distribution of Array shape axis 1')

# Array shape axis 2
plt.subplot(1, 3, 3)
sns.histplot(df['Array shape (axis 2)'], kde=True)
plt.title('Distribution of Array shape axis 2')

plt.tight_layout()
plt.show()


# Visualize the distribution of voxel spacing and number of motors
plt.figure(figsize=(10, 5))

# Voxel spacing
plt.subplot(1, 2, 1)
sns.histplot(df['Voxel spacing'], kde=True)
plt.title('Distribution of Voxel Spacing')

# Number of motors
plt.subplot(1, 2, 2)
sns.histplot(df['Number of motors'], kde=True)
plt.title('Distribution of Number of Motors')

plt.tight_layout()
plt.show()


# Bar plot for the number of tomograms with motors vs. without
plt.figure(figsize=(6, 4))

# Count the number of tomograms with and without motors
motor_counts = df['Number of motors'].apply(lambda x: 'With Motors' if x > 0 else 'Without Motors').value_counts()

# Plot the bar chart
sns.barplot(x=motor_counts.index, y=motor_counts.values)
plt.title('Distribution of Tomograms: With Motors vs. Without Motors')
plt.xlabel('Tomogram Type')
plt.ylabel('Count')

plt.show()


import os
from PIL import Image
import random

# Get the first tomo_id from our dataset
first_tomo = df['tomo_id'].iloc[1]

# Construct the base path for the tomogram
tomo_path = f'{train_dir}/{first_tomo}'

# Get list of all slice files in the directory
slice_files = [f for f in os.listdir(tomo_path) if f.startswith('slice_') and f.endswith('.jpg')]

# Randomly select 9 slices
selected_slices = random.sample(slice_files, min(9, len(slice_files)))

# Create a 3x3 grid of subplots
fig, axes = plt.subplots(3, 3, figsize=(15, 15))
fig.suptitle(f'Sample Slices from {first_tomo}')

# Load and display each image
for idx, slice_file in enumerate(selected_slices):
    row = idx // 3
    col = idx % 3
    
    # Load image
    img_path = os.path.join(tomo_path, slice_file)
    img = Image.open(img_path)
    
    # Display image
    axes[row, col].imshow(img, cmap='gray')
    axes[row, col].axis('off')
    axes[row, col].set_title(f'Slice {slice_file[6:-4]}')

plt.tight_layout()
plt.show()

# print(f"Displaying 9 random slices from tomogram: {first_tomo}")


def visualize_motor_location(index):
    row = df.iloc[index]
    folder = f'{train_dir}/{row.tomo_id}'
    motor_slice = int(row['Motor axis 0'])

    # Add sufficient zeros to make it a 4-digit integer
    motor_slice = str(motor_slice).zfill(4)

    print(f'Motor at slice: {motor_slice}')
    slice_file = f'{folder}/slice_{motor_slice}.jpg'

    # Open the image
    img = Image.open(slice_file).convert('L')  # Convert to grayscale

    # Get the motor 1 and 2 axis values
    motor1 = int(row['Motor axis 1'])
    motor2 = int(row['Motor axis 2'])

    # Draw a 50% transparent red circle on the img around the motor1, motor2 position with a 50-pixel radius
    fig, ax = plt.subplots()
    ax.imshow(img, cmap='gray')
    ax.add_artist(plt.Circle((motor2, motor1), 50, color='r', alpha=0.5))
    ax.axis('off')
    plt.show()

# Example usage
visualize_motor_location(1)


import matplotlib.animation as animation
from IPython.display import HTML
import matplotlib as mpl

# Increase the animation embed limit
mpl.rcParams['animation.embed_limit'] = 50  # Set to 60 MB

def animate_tomogram(tomo_id):
    folder = f'{train_dir}/{tomo_id}'
    slice_files = sorted([f for f in os.listdir(folder) if f.startswith('slice_') and f.endswith('.jpg')])

    # Load images into a 3D array
    images = [np.array(Image.open(os.path.join(folder, f)).convert('L')) for f in slice_files]

    fig, ax = plt.subplots()
    img_display = ax.imshow(images[0], cmap='gray')
    ax.axis('off')

    def update(frame):
        img_display.set_array(images[frame])
        return [img_display]

    # Reduce the number of frames by skipping some
    ani = animation.FuncAnimation(fig, update, frames=range(0, len(images), 2), interval=100, blit=True)

    # Add start/stop buttons and a slider
    plt.close(fig)
    return HTML(ani.to_jshtml())

# run animation
#animate_tomogram('tomo_00e047')


# read sumbission file
ss = pd.read_csv(f'{base_path}/sample_submission.csv')

 # remove the -1's and get the mean values of the motor positions
motor_positions = df[df['Motor axis 0'] != -1][['Motor axis 0', 'Motor axis 1', 'Motor axis 2']]
motor_positions_mean = motor_positions.mean()

ss['Motor axis 0'] = motor_positions_mean.iloc[0]
ss['Motor axis 1'] = motor_positions_mean.iloc[1]
ss['Motor axis 2'] = motor_positions_mean.iloc[2]

ss.to_csv('submission.csv',index=False)





