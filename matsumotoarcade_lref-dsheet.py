# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session





import pandas as pd
import math

def calculate_lref_d0(df):
  """Calculates Lref and d0 for each entry in the DataFrame.

  Args:
    df: Pandas DataFrame containing the data.

  Returns:
    Pandas DataFrame with added 'Lref' and 'd0' columns.
  """
  df['Lref'] = df['sequence'].apply(len)
  df['d0'] = df['Lref'].apply(lambda lref: 0.6 * math.sqrt(lref - 0.5) - 2.5)
  return df

# Load the CSV files
try:
    validation_df = pd.read_csv("validation_sequences.csv")
    test_df = pd.read_csv("test_sequences.csv")
except FileNotFoundError:
    print("Error: validation_sequences.csv or test_sequences.csv not found.  Make sure the files are in the same directory as the script.")
    # Create empty DataFrames to prevent NameError, but analysis will be limited.
    validation_df = pd.DataFrame()
    test_df = pd.DataFrame()
    # Exit or continue with empty dataframes.  For Kaggle, it's better to continue
    # so the notebook still runs, even if it doesn't do much.
    # exit()  # Uncomment if you want to stop completely on FileNotFoundError
    print("Continuing with empty DataFrames.")


# Calculate Lref and d0 for both DataFrames
if not validation_df.empty:  # Only process if the DataFrame is not empty
    validation_df = calculate_lref_d0(validation_df)
    print("Validation Data:")
    print(validation_df[['target_id', 'sequence', 'Lref', 'd0']]) # Showing only relevant columns
else:
    print("Validation DataFrame is empty, skipping Lref/d0 calculation.")

if not test_df.empty:  # Only process if the DataFrame is not empty
    test_df = calculate_lref_d0(test_df)
    print("\nTest Data:")
    print(test_df[['target_id', 'sequence', 'Lref', 'd0']]) # Showing only relevant columns
else:
    print("Test DataFrame is empty, skipping Lref/d0 calculation.")



# Optionally, save the updated DataFrames to new CSV files:
# validation_df.to_csv("validation_sequences_with_lref_d0.csv", index=False)
# test_df.to_csv("test_sequences_with_lref_d0.csv", index=False)


import os

base_path = "/kaggle/input/stanford-rna-3d-folding/"

validation_csv_path = os.path.join(base_path, "validation_sequences.csv")
test_csv_path = os.path.join(base_path, "test_sequences.csv")

print(f"Validation CSV Path: {validation_csv_path}")
print(f"Test CSV Path: {test_csv_path}")

print(f"Validation CSV Exists: {os.path.exists(validation_csv_path)}")
print(f"Test CSV Exists: {os.path.exists(test_csv_path)}")


import pandas as pd
import math
import os

def calculate_lref_d0(df):
  """Calculates Lref and d0 for each entry in the DataFrame.

  Args:
    df: Pandas DataFrame containing the data.

  Returns:
    Pandas DataFrame with added 'Lref' and 'd0' columns.
  """
  df['Lref'] = df['sequence'].apply(len)
  df['d0'] = df['Lref'].apply(lambda lref: 0.6 * math.sqrt(lref - 0.5) - 2.5)
  return df

# Define the base path for the input files in Kaggle
competition_name = "stanford-rna-3d-folding"  # Replace with the actual competition name
base_path = f"/kaggle/input/{competition_name}/"

# Load the CSV files
validation_csv_path = os.path.join(base_path, "validation_sequences.csv")
test_csv_path = os.path.join(base_path, "test_sequences.csv")

# The files exist, so we can load them directly
validation_df = pd.read_csv(validation_csv_path)
test_df = pd.read_csv(test_csv_path)


# Calculate Lref and d0 for both DataFrames
validation_df = calculate_lref_d0(validation_df)
print("Validation Data:")
print(validation_df[['target_id', 'sequence', 'Lref', 'd0']]) # Showing only relevant columns


test_df = calculate_lref_d0(test_df)
print("\nTest Data:")
print(test_df[['target_id', 'sequence', 'Lref', 'd0']]) # Showing only relevant columns



# Optionally, save the updated DataFrames to new CSV files:
# validation_df.to_csv("validation_sequences_with_lref_d0.csv", index=False)
# test_df.to_csv("test_sequences_with_lref_d0.csv", index=False)


import pandas as pd
import math
import os
from IPython.display import display, HTML
import numpy as np  # Import numpy for creating sample predictions

def calculate_lref_d0(df):
    """Calculates Lref and d0 for each entry in the DataFrame."""
    df['Lref'] = df['sequence'].apply(len)
    df['d0'] = df['Lref'].apply(lambda lref: 0.6 * math.sqrt(lref - 0.5) - 2.5)
    return df

# Define the base path for the input files in Kaggle
competition_name = "stanford-rna-3d-folding"
base_path = f"/kaggle/input/{competition_name}/"

# Load the CSV files
validation_csv_path = os.path.join(base_path, "validation_sequences.csv")
test_csv_path = os.path.join(base_path, "test_sequences.csv")

# The files exist, so we can load them directly
validation_df = pd.read_csv(validation_csv_path)
test_df = pd.read_csv(test_csv_path)

# Calculate Lref and d0 for both DataFrames
validation_df = calculate_lref_d0(validation_df)
print("<h2>Validation Data:</h2>")
display(HTML(validation_df[['target_id', 'sequence', 'Lref', 'd0']].to_html(index=False)))

test_df = calculate_lref_d0(test_df)
print("<h2>Test Data:</h2>")
display(HTML(test_df[['target_id', 'sequence', 'Lref', 'd0']].to_html(index=False)))


# --- PREDICTION CODE STARTS HERE ---
# REPLACE THIS SECTION WITH YOUR ACTUAL MODEL AND PREDICTION LOGIC
# This is placeholder code to generate sample predictions.  You must adapt this
# to use your trained model to generate predictions based on the sequences in test_df.
# Make sure that 'predictions' is a Pandas Series or NumPy array with one prediction
# for each row in 'test_df'.
np.random.seed(42)  # for reproducibility
predictions = np.random.rand(len(test_df))
# --- PREDICTION CODE ENDS HERE ---


# --- CREATE SUBMISSION FILE ---
submission = pd.DataFrame({'target_id': test_df['target_id'], 'predicted_value': predictions})
submission['predicted_value'] = submission['predicted_value'].astype(float) #Ensure its numeric

submission_filename = 'submission.csv'
submission.to_csv(submission_filename, index=False)


# --- VERIFY SUBMISSION FILE CREATION ---
if os.path.exists(submission_filename):
    print(f"Submission file '{submission_filename}' created successfully!")
    print("\nFirst 5 rows of submission.csv:")
    display(HTML(submission.head().to_html(index=False))) # Show the first few rows
else:
    print(f"ERROR: Submission file '{submission_filename}' was NOT created!")

print("Make sure to submit file named submission.csv")


import pandas as pd
import math
import os
from IPython.display import display, HTML
import numpy as np
import matplotlib.pyplot as plt  # Import for plotting
from mpl_toolkits.mplot3d import Axes3D  # Import for 3D plotting

# Assume the existence of the following file on the host:
#   - test_sequences.csv: test sequences

def calculate_lref_d0(df):
    """Calculates Lref and d0 for each entry in the DataFrame."""
    df['Lref'] = df['sequence'].apply(len)
    df['d0'] = df['Lref'].apply(lambda lref: 0.6 * math.sqrt(lref - 0.5) - 2.5)
    return df

# Define the base path for the input files in Kaggle
competition_name = "stanford-rna-3d-folding"
base_path = f"/kaggle/input/{competition_name}/"

# Load the CSV files
try:
    validation_csv_path = os.path.join(base_path, "validation_sequences.csv")
    test_csv_path = os.path.join(base_path, "test_sequences.csv")

    # Load the dataframes, raising FileNotFoundError exception if the files can't be found.
    validation_df = pd.read_csv(validation_csv_path)
    test_df = pd.read_csv(test_csv_path)
except FileNotFoundError as e:
    print(f"Error: {e}")
    #If either of the dataframes can't be found, stop execution.
    raise

# Calculate Lref and d0 for both DataFrames
validation_df = calculate_lref_d0(validation_df)
print("<h2>Validation Data:</h2>")
display(HTML(validation_df[['target_id', 'sequence', 'Lref', 'd0']].to_html(index=False)))

test_df = calculate_lref_d0(test_df)
print("<h2>Test Data:</h2>")
display(HTML(test_df[['target_id', 'sequence', 'Lref', 'd0']].to_html(index=False)))

# --- PREDICTION CODE STARTS HERE ---
# REPLACE THIS ENTIRE SECTION WITH YOUR ACTUAL MODEL AND 3D STRUCTURE PREDICTION LOGIC!
# This is placeholder code to generate random coordinates, attempting to simulate
# a helical structure. You MUST adapt this to use your trained model to predict
# the 3D structure of each RNA sequence in 'test_df'.

# Parameters to control the helix generation (adjust these!)
num_structures = 1 # Reduced to 1 to make visualization clearer
np.random.seed(42)  # For reproducibility
helix_radius = 5.0  # Radius of the helix
helix_pitch = 3.0  # Vertical distance between turns
z_scaling_factor = 1.0

#Define base_points - Use the first entry as the start to the transformation
base_point_x = np.array(test_df['Lref'])[0]
base_point_y = np.array(test_df['d0'])[0]
base_points = np.array([base_point_x, base_point_y])

def k_vector_field(x, structure_number):
    """Defines a simple k-vector field"""
    # This is a very simple example: a constant vector field
    # Scale based on how far the structure is from the origin
    scaling = 0.1 * structure_number
    return np.array([scaling, scaling])

def integral_section(base_point, k_vector_field, sequence_length, structure_number):
    """Applies the integral section to create the 3d points, transforming the points"""
    angles = np.linspace(0, 4 * np.pi, sequence_length)  # Angles for points along the helix (4 pi = two full rotations)
    random_offsets = np.random.rand(sequence_length) * 1.0  # Add some random noise

    #Apply transformation based on vector field
    vector_field = k_vector_field(base_point, structure_number) # Scale the transformation based on the vectorfield
    x = helix_radius * np.cos(angles) + random_offsets + vector_field[0]
    y = helix_radius * np.sin(angles) + random_offsets + vector_field[1]
    z = helix_pitch * angles  + z_scaling_factor * structure_number #Linear increase + scalingFactor
    return x,y,z

def generate_helical_structure(sequence_length, structure_number):
    """Generates a single helical 3D structure (coordinates for each residue)."""
    angles = np.linspace(0, 4 * np.pi, sequence_length)  # Angles for points along the helix (4 pi = two full rotations)
    random_offsets = np.random.rand(sequence_length) * 1.0  # Add some random noise
    x_coords = helix_radius * np.cos(angles) + random_offsets
    y_coords = helix_radius * np.sin(angles) + random_offsets
    z_coords = helix_pitch * angles  + z_scaling_factor * structure_number
    return x_coords, y_coords, z_coords

all_structure_data = []
for index, row in test_df.iterrows():
    target_id = row['target_id']
    sequence = row['sequence']
    sequence_length = len(sequence)

    # Generate a color for the sequence (for plotting)
    color = np.random.rand(3,)  # Generate a random RGB color
    # plot with color of choice
    fig = plt.figure(figsize=(8, 8))  # Create a figure
    ax = fig.add_subplot(111, projection='3d')  # Add a 3D subplot

    for structure_num in range(1, num_structures + 1):
        x_coords, y_coords, z_coords = integral_section(base_points, k_vector_field, sequence_length, structure_num)

        # Plot the helical structure with a custom color
        ax.plot(x_coords, y_coords, z_coords, c=color, label=f"{target_id}")  # Change from scatter to plot, for line

        for residue_index in range(sequence_length):
            resname = sequence[residue_index]  # Get the residue name from the sequence
            resid = residue_index + 1  # Residue ID (1-based)
            structure_id = f"{target_id}_{structure_num}"

            # Create structure for the current structure
            all_structure_data.append([structure_id, resname, resid,
                                        x_coords[residue_index],
                                        y_coords[residue_index],
                                        z_coords[residue_index]])

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"3D Structure of {target_id}") # Title with id
    ax.legend()  # Show legend to identify the structures

    #Adjust axes ranges
    max_range = np.array([x_coords.max()-x_coords.min(), y_coords.max()-y_coords.min(), z_coords.max()-z_coords.min()]).max() / 2.0
    mid_x = (x_coords.max()+x_coords.min()) * 0.5
    mid_y = (y_coords.max()+y_coords.min()) * 0.5
    mid_z = (z_coords.max()+z_coords.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.show()  # Display the plot

#Create the columns based on the structure of all_structure_data
col_names = ['ID', 'resname', 'resid', 'x_1', 'y_1', 'z_1']

# --- PREDICTION CODE ENDS HERE ---

# --- CREATE SUBMISSION FILE ---
submission_df = pd.DataFrame(all_structure_data, columns=col_names)

submission_filename = 'submission.csv'
submission_df.to_csv(submission_filename, index=False)

# --- VERIFY SUBMISSION FILE CREATION ---
if os.path.exists(submission_filename):
    print(f"Submission file '{submission_filename}' created successfully!")
    print("\nFirst 5 rows of submission.csv:")
    display(HTML(submission_df.head().to_html(index=False)))  # Show the first few rows
    print(f"\nSubmission contains: {len(submission_df)} rows")  # Shows the number of rows
else:
    print(f"ERROR: Submission file '{submission_filename}' was NOT created!")

print("Make sure to submit file named submission.csv")


import pandas as pd
import math
import os
from IPython.display import display, HTML
import numpy as np
import matplotlib.pyplot as plt  # Import for plotting
from mpl_toolkits.mplot3d import Axes3D  # Import for 3D plotting

# Assume the existence of the following file on the host:
#   - test_sequences.csv: test sequences

def calculate_lref_d0(df):
    """Calculates Lref and d0 for each entry in the DataFrame."""
    df['Lref'] = df['sequence'].apply(len)
    df['d0'] = df['Lref'].apply(lambda lref: 0.6 * math.sqrt(lref - 0.5) - 2.5)
    return df

# Define the base path for the input files in Kaggle
competition_name = "stanford-rna-3d-folding"
base_path = f"/kaggle/input/{competition_name}/"

# Load the CSV files
try:
    validation_csv_path = os.path.join(base_path, "validation_sequences.csv")
    test_csv_path = os.path.join(base_path, "test_sequences.csv")

    # Load the dataframes, raising FileNotFoundError exception if the files can't be found.
    validation_df = pd.read_csv(validation_csv_path)
    test_df = pd.read_csv(test_csv_path)
except FileNotFoundError as e:
    print(f"Error: {e}")
    #If either of the dataframes can't be found, stop execution.
    raise

# Calculate Lref and d0 for both DataFrames
validation_df = calculate_lref_d0(validation_df)
print("<h2>Validation Data:</h2>")
display(HTML(validation_df[['target_id', 'sequence', 'Lref', 'd0']].to_html(index=False)))

test_df = calculate_lref_d0(test_df)
print("<h2>Test Data:</h2>")
display(HTML(test_df[['target_id', 'sequence', 'Lref', 'd0']].to_html(index=False)))

# --- PREDICTION CODE STARTS HERE ---
# REPLACE THIS ENTIRE SECTION WITH YOUR ACTUAL MODEL AND 3D STRUCTURE PREDICTION LOGIC!
# This is placeholder code to generate random coordinates, attempting to simulate
# a helical structure. You MUST adapt this to use your trained model to predict
# the 3D structure of each RNA sequence in 'test_df'.

# Parameters to control the helix generation (adjust these!)
num_structures = 1
np.random.seed(42)  # For reproducibility
helix_radius = 5.0  # Radius of the helix
helix_pitch = 3.0  # Vertical distance between turns
z_scaling_factor = 1.0

def generate_helical_structure(sequence_length, structure_number):
    """Generates a single helical 3D structure (coordinates for each residue)."""
    angles = np.linspace(0, 4 * np.pi, sequence_length)  # Angles for points along the helix (4 pi = two full rotations)
    random_offsets = np.random.rand(sequence_length) * 1.0  # Add some random noise
    x_coords = helix_radius * np.cos(angles) + random_offsets
    y_coords = helix_radius * np.sin(angles) + random_offsets
    z_coords = helix_pitch * angles  + z_scaling_factor * structure_number
    return x_coords, y_coords, z_coords

all_structure_data = []

# Create a single figure and axes for all plots
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("3D Structures of RNA Sequences")  # General plot title

# Store the initial point
initial_point = np.array([0,0,0]) #Origin (Will shift to be sequence base)

# Generate the plots together
for index, row in test_df.iterrows():
    target_id = row['target_id']
    sequence = row['sequence']
    sequence_length = len(sequence)

    # Generate a color for the sequence (for plotting)
    color = np.random.rand(3,)  # Generate a random RGB color

    for structure_num in range(1, num_structures + 1):
        x_coords, y_coords, z_coords = generate_helical_structure(sequence_length, structure_num)

        # Applying φ(0) = x for point
        starting_point = np.array([x_coords[0], y_coords[0], z_coords[0]]) # The point the function has been derived to get to
        x_coords = x_coords - starting_point[0]
        y_coords = y_coords - starting_point[1]
        z_coords = z_coords - starting_point[2] # Apply to the coordinate system

        # Applying φ∗(t) to generate lines through those point
        #Since all values have been offset, now connect to origin (base points)
        ax.plot([initial_point[0]],[initial_point[1]],[initial_point[2]], 'o', c=color, markersize=4) #Point
        ax.plot(x_coords, y_coords, z_coords, c=color, label=f"{target_id}") #

        for residue_index in range(sequence_length):
            resname = sequence[residue_index]  # Get the residue name from the sequence
            resid = residue_index + 1  # Residue ID (1-based)
            structure_id = f"{target_id}_{structure_num}"

            # Create structure for the current structure
            all_structure_data.append([structure_id, resname, resid,
                                        x_coords[residue_index],
                                        y_coords[residue_index],
                                        z_coords[residue_index]])

ax.legend() # Add a legend
plt.show() # Show plots

#Create the columns based on the structure of all_structure_data
col_names = ['ID', 'resname', 'resid', 'x_1', 'y_1', 'z_1']

# --- PREDICTION CODE ENDS HERE ---

# --- CREATE SUBMISSION FILE ---
submission_df = pd.DataFrame(all_structure_data, columns=col_names)

submission_filename = 'submission.csv'
submission_df.to_csv(submission_filename, index=False)

# --- VERIFY SUBMISSION FILE CREATION ---
if os.path.exists(submission_filename):
    print(f"Submission file '{submission_filename}' created successfully!")
    print("\nFirst 5 rows of submission.csv:")
    display(HTML(submission_df.head().to_html(index=False)))  # Show the first few rows
    print(f"\nSubmission contains: {len(submission_df)} rows")  # Shows the number of rows
else:
    print(f"ERROR: Submission file '{submission_filename}' was NOT created!")

print("Make sure to submit file named submission.csv")





import pandas as pd
import math
import os
from IPython.display import display, HTML
import numpy as np
import matplotlib.pyplot as plt  # Import for plotting
from mpl_toolkits.mplot3d import Axes3D  # Import for 3D plotting

# Assume the existence of the following file on the host:
#   - test_sequences.csv: test sequences

def calculate_lref_d0(df):
    """Calculates Lref and d0 for each entry in the DataFrame."""
    df['Lref'] = df['sequence'].apply(len)
    df['d0'] = df['Lref'].apply(lambda lref: 0.6 * math.sqrt(lref - 0.5) - 2.5)
    return df

# Define the base path for the input files in Kaggle
competition_name = "stanford-rna-3d-folding"
base_path = f"/kaggle/input/{competition_name}/"

# Load the CSV files
try:
    validation_csv_path = os.path.join(base_path, "validation_sequences.csv")
    test_csv_path = os.path.join(base_path, "test_sequences.csv")

    # Load the dataframes, raising FileNotFoundError exception if the files can't be found.
    validation_df = pd.read_csv(validation_csv_path)
    test_df = pd.read_csv(test_csv_path)
except FileNotFoundError as e:
    print(f"Error: {e}")
    #If either of the dataframes can't be found, stop execution.
    raise

# Calculate Lref and d0 for both DataFrames
validation_df = calculate_lref_d0(validation_df)
print("<h2>Validation Data:</h2>")
display(HTML(validation_df[['target_id', 'sequence', 'Lref', 'd0']].to_html(index=False)))

test_df = calculate_lref_d0(test_df)
print("<h2>Test Data:</h2>")
display(HTML(test_df[['target_id', 'sequence', 'Lref', 'd0']].to_html(index=False)))

# --- PREDICTION CODE STARTS HERE ---
# REPLACE THIS ENTIRE SECTION WITH YOUR ACTUAL MODEL AND 3D STRUCTURE PREDICTION LOGIC!
# This is placeholder code to generate random coordinates, attempting to simulate
# a helical structure. You MUST adapt this to use your trained model to predict
# the 3D structure of each RNA sequence in 'test_df'.

# Parameters to control the helix generation (adjust these!)
num_structures = 2 # Generate two structures (double helix)
np.random.seed(42)  # For reproducibility
helix_radius = 5.0  # Radius of the helix
helix_pitch = 3.0  # Vertical distance between turns
z_scaling_factor = 1.0

#Define base_points - Use the first entry as the start to the transformation
base_point_x = np.array(test_df['Lref'])[0]
base_point_y = np.array(test_df['d0'])[0]
base_points = np.array([base_point_x, base_point_y])

def k_vector_field(x, structure_number):
    """Defines a simple k-vector field"""
    # This is a very simple example: a constant vector field
    # Scale based on how far the structure is from the origin
    scaling = 0.1 * structure_number
    return np.array([scaling, scaling])

def integral_section(base_point, k_vector_field, sequence_length, structure_number):
    """Applies the integral section to create the 3d points, transforming the points"""
    angles = np.linspace(0, 4 * np.pi, sequence_length)  # Angles for points along the helix (4 pi = two full rotations)
    random_offsets = np.random.rand(sequence_length) * 1.0  # Add some random noise

    # Introduce a phase shift for the second helix
    phase_shift = np.pi if structure_number == 2 else 0  # 180-degree phase shift

    #Apply transformation based on vector field
    vector_field = k_vector_field(base_point, structure_number) # Scale the transformation based on the vectorfield
    x = helix_radius * np.cos(angles + phase_shift) + random_offsets + vector_field[0]
    y = helix_radius * np.sin(angles + phase_shift) + random_offsets + vector_field[1]
    z = helix_pitch * angles  + z_scaling_factor * structure_number #Linear increase + scalingFactor
    return x,y,z

def generate_helical_structure(sequence_length, structure_number):
    """Generates a single helical 3D structure (coordinates for each residue)."""
    angles = np.linspace(0, 4 * np.pi, sequence_length)  # Angles for points along the helix (4 pi = two full rotations)
    random_offsets = np.random.rand(sequence_length) * 1.0  # Add some random noise
    x_coords = helix_radius * np.cos(angles) + random_offsets
    y_coords = helix_radius * np.sin(angles) + random_offsets
    z_coords = helix_pitch * angles  + z_scaling_factor * structure_number
    return x_coords, y_coords, z_coords

all_structure_data = []
for index, row in test_df.iterrows():
    target_id = row['target_id']
    sequence = row['sequence']
    sequence_length = len(sequence)

    # Generate a color for the sequence (for plotting)
    # color = np.random.rand(3,)  # Generate a random RGB color #Removed random color, assigning one to the structure instead.

    # plot with color of choice
    fig = plt.figure(figsize=(8, 8))  # Create a figure
    ax = fig.add_subplot(111, projection='3d')  # Add a 3D subplot

    for structure_num in range(1, num_structures + 1):

        # Assign different color per structure to differentiate double helix
        color = 'red' if structure_num == 1 else 'blue'

        x_coords, y_coords, z_coords = integral_section(base_points, k_vector_field, sequence_length, structure_num)

        # Plot the helical structure with a custom color
        ax.plot(x_coords, y_coords, z_coords, c=color, label=f"{target_id} Helix {structure_num}")  # Change from scatter to plot, for line

        for residue_index in range(sequence_length):
            resname = sequence[residue_index]  # Get the residue name from the sequence
            resid = residue_index + 1  # Residue ID (1-based)
            structure_id = f"{target_id}_{structure_num}"

            # Create structure for the current structure
            all_structure_data.append([structure_id, resname, resid,
                                        x_coords[residue_index],
                                        y_coords[residue_index],
                                        z_coords[residue_index]])

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"3D Structure of {target_id}") # Title with id
    ax.legend()  # Show legend to identify the structures

    #Adjust axes ranges
    max_range = np.array([x_coords.max()-x_coords.min(), y_coords.max()-y_coords.min(), z_coords.max()-z_coords.min()]).max() / 2.0
    mid_x = (x_coords.max()+x_coords.min()) * 0.5
    mid_y = (y_coords.max()+y_coords.min()) * 0.5
    mid_z = (z_coords.max()+z_coords.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.show()  # Display the plot

#Create the columns based on the structure of all_structure_data
col_names = ['ID', 'resname', 'resid', 'x_1', 'y_1', 'z_1']

# --- PREDICTION CODE ENDS HERE ---

# --- CREATE SUBMISSION FILE ---
submission_df = pd.DataFrame(all_structure_data, columns=col_names)

submission_filename = 'submission.csv'
submission_df.to_csv(submission_filename, index=False)

# --- VERIFY SUBMISSION FILE CREATION ---
if os.path.exists(submission_filename):
    print(f"Submission file '{submission_filename}' created successfully!")
    print("\nFirst 5 rows of submission.csv:")
    display(HTML(submission_df.head().to_html(index=False)))  # Show the first few rows
    print(f"\nSubmission contains: {len(submission_df)} rows")  # Shows the number of rows
else:
    print(f"ERROR: Submission file '{submission_filename}' was NOT created!")

print("Make sure to submit file named submission.csv")


import pandas as pd
import math
import os
from IPython.display import display, HTML
import numpy as np

def generate_human_dna_helix(sequence_length, structure_number):
    """Generates a single helical 3D structure simulating human DNA base composition.

    Args:
      sequence_length:  The length of the DNA sequence to generate
      structure_number: A parameter to slightly vary the helix for multiple structures.

    Returns:
      x_coords, y_coords, z_coords:  NumPy arrays of coordinates.
    """
    #Base percentages of A,T,G,C
    a_percent = 0.3
    t_percent = 0.303
    g_percent = 0.195
    c_percent = 0.199

    base_composition = {'A': a_percent, 'T': t_percent, 'G': g_percent, 'C': c_percent}

    # Normalize probabilities to sum to 1
    total_probability = sum(base_composition.values())
    for base in base_composition:
        base_composition[base] /= total_probability

    #Generate the base code
    dna_sequence = ''.join(np.random.choice(list(base_composition.keys()), sequence_length, p=list(base_composition.values())))
    #print (f"Here is the {sequence_length} DNA base code: {dna_sequence}") #Unnecessary print
    helix_radius = 5.0
    helix_pitch = 3.0
    angles = np.linspace(0, 4 * np.pi, sequence_length)  # Helix angles (4 pi = two full turns)
    random_offsets = np.random.rand(sequence_length) # Add some noise

    x_coords = helix_radius * np.cos(angles) + random_offsets
    y_coords = helix_radius * np.sin(angles) + random_offsets
    z_coords = helix_pitch * angles  # Linear increase for the helix, multiplied by the structure number

    return dna_sequence, x_coords, y_coords, z_coords #Return the bases too, for each dna structure


def generate_submission_file(test_file, num_structures = 5):
    """Generates submission file for the competition, as described in problem desciption"""

    # Load data:
    test_df = pd.read_csv(test_file)

    # Prepare a list to hold all points (for the submission file)
    submission_list = [] # The data

    #Create a total of num_structures coordinates per item, and add it to the file
    for num_row_test in range(test_df.shape[0]):
        row_test = test_df.iloc[num_row_test, :] #Extract the test data
        target_id = row_test["target_id"] #What the results get mapped to

        sequence = row_test["sequence"]  # What data gets run through
        total_pts = len(sequence)        # What the sequence will be

        for number_runs in range(num_structures):   # Create all combinations for each target_id

            #Run the results through your algorithm, and store the data points.
            bases, x_coords, y_coords, z_coords = generate_human_dna_helix(total_pts, number_runs + 1) # run the code for a

            for res_num in range (total_pts):  # Store all values to the list

                #Create a unique targetID from the run.
                unique_id = target_id + "_" + str(number_runs + 1) #Each run gets added

                #Make sure there are the right datatypes, since there have been many errors related to this.
                data_row = [unique_id,  str(bases[res_num]), str(res_num + 1), float(x_coords[res_num]), float(y_coords[res_num]), float(z_coords[res_num])]

                #Append
                submission_list.append(data_row)

    #Name headers that are stored
    header_names = ["ID", "resname", "resid", "x_1", "y_1", "z_1"]

    #Create the Dataframe from the stored file.
    df = pd.DataFrame(submission_list, columns=header_names)

    #Create and export the dataframe to csv.
    df.to_csv("submission.csv", index = False)

    #Load df.
    submission = pd.read_csv("submission.csv") #Ensure that datatypes are what is expected
    print (submission.head(50))
    return submission

#Replace this filename to read through the submission algorithm.
competition_name = "stanford-rna-3d-folding"
filename = f"/kaggle/input/{competition_name}/test_sequences.csv" #Added in filename, the directory

submission = generate_submission_file(filename)

print ("Finished!")


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def generate_double_helix(sequence_length, structure_number):
    """Generates a single helical 3D structure, as well as its partner helix.

    Args:
      sequence_length:  The length of the DNA sequence to generate
      structure_number: A parameter to slightly vary the helix for multiple structures.

    Returns:
      x_coords, y_coords, z_coords:  Arrays of coordinates for helix 1.
      x_coords2, y_coords2, z_coords2:  Arrays of coordinates for helix 2.
    """
    helix_radius = 5.0
    helix_pitch = 3.0
    angles = np.linspace(0, 4 * np.pi, sequence_length)  # Helix angles (4 pi = two full turns)
    random_offsets = np.random.rand(sequence_length) # Add some noise

    #Helix 1
    x_coords = helix_radius * np.cos(angles) + random_offsets
    y_coords = helix_radius * np.sin(angles) + random_offsets
    z_coords = helix_pitch * angles  # Linear increase for the helix

    #Helix 2 (Partner)
    x_coords2 = helix_radius * np.cos(angles + np.pi) + random_offsets #Phase shift
    y_coords2 = helix_radius * np.sin(angles + np.pi) + random_offsets #Phase shift
    z_coords2 = helix_pitch * angles #The code will run with the structure number, and it is not required in current use

    return x_coords, y_coords, z_coords, x_coords2, y_coords2, z_coords2

sequence_length = 100 #Number points in sequence, adjust
num_structure = 1

#Example of code running:
#Generate helices for plot
x_coords, y_coords, z_coords, x_coords2, y_coords2, z_coords2 = generate_double_helix(sequence_length, num_structure)

#Set up figure for plot, with all of the required axis
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

#Plot the helixes
ax.plot(x_coords, y_coords, z_coords, c='blue', label="Helix 1") #Sets the single point as the start
ax.plot(x_coords2, y_coords2, z_coords2, c='red', label="Helix 2") #Sets the single point as the start

#Plot all graph information to setup plot
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Double Helix Structure")
ax.legend()

#Set a limit that is equal to all of the points to generate structure
x_range = np.abs(x_coords).max()
y_range = np.abs(y_coords).max()
z_range = np.abs(z_coords).max()
max_range = np.array([x_range, y_range, z_range]).max()

ax_range = 1.2*(max_range + 1)
ax.set_xlim([-ax_range, ax_range])
ax.set_ylim([-ax_range, ax_range])
ax.set_zlim([0, ax_range])

#Rotate camera as needed
plt.show()


import pandas as pd
import numpy as np
from itertools import combinations

def find_best_double_helices(submission_file):
    """
    Finds the best-matching pairs of helices from a submission file based on a similarity score.

    Args:
      submission_file: The path to the submission.csv file.

    Returns:
      A Pandas DataFrame with the best double helix pairs and their similarity scores.
    """
    try:
        submission = pd.read_csv(submission_file)
    except FileNotFoundError:
        print(f"Error: Submission file '{submission_file}' not found.")
        return None

    # 1. Group by Target ID
    submission['structure_num'] = submission['ID'].apply(lambda x: int(x.split('_')[-1]))  # Extract the structure number
    grouped = submission.groupby('ID')

    # 2.  Extract the coordinate data for each coordinate
    helix_data = {} #Store list
    for name, group in grouped:
        coordinates = group[['x_1', 'y_1', 'z_1']].values
        helix_data[name] = coordinates

    # 3. Code to test how similar a dataset is
    def calculate_similarity(helix1, helix2):
        """Calculates a similarity score between two helices."""
        diff = np.abs(helix1 - helix2) #Differences for all values
        similarity = np.sum(diff) #Reduce all to single point
        return similarity

    # 4. Determine each sequence combination.
    helix_ids = list(helix_data.keys())  # Get a list of all helix IDs.
    helix_pairs = list(combinations(helix_ids, 2)) #Test all unique pairings


    # 5. Find the best pairing
    best_pairs = []  # store the results
    all_used_ids = [] # All used combinations.
    for target_id in test_df['target_id']: #Each unique target
        possible_combinations = []  # all combos
        target_combos = [pair for pair in helix_pairs if target_id in pair[0] and target_id in pair[1]] #Find combinations
        target_combos = []
        for pair in helix_pairs: #For all the helical pairs
            if target_id in pair[0] and target_id in pair[1]:
                for check_used in all_used_ids:
                  if pair == check_used:
                      break #If they've been used, don't generate point
                target_combos.append(pair) #Else, add combination
                all_used_ids.append(pair) #Added

        best_pair = None
        min_similarity = float('inf') # High to mean will always be smaller.
        for pair in target_combos: # Test the current target's combo
            #Get the id of the target.
            helix1_id, helix2_id = pair
            helix1 = helix_data[helix1_id] # Test that target ID
            helix2 = helix_data[helix2_id] # Test that target ID

            similarity = calculate_similarity(helix1, helix2) # Test how different they are
            if similarity < min_similarity:
                min_similarity = similarity
                best_pair = (helix1_id, helix2_id) # Set that point as smallest

        best_pairs.append((best_pair[0], best_pair[1], min_similarity)) # All smallest points are stored

    # 6. All results of the best matches gets put into dataframe
    double_helices = []
    for helix1_id, helix2_id, similarity in best_pairs:
        double_helices.append([helix1_id, helix2_id, similarity])
        print (f"Helix pairing best match: {helix1_id} and {helix2_id}, Similarity: {similarity}")

    #Create a Dataframe from the stored file, will then be used for plot.
    best_pairings = pd.DataFrame(double_helices, columns=['Helix 1', 'Helix 2', 'Similarity'])
    return best_pairings

# Test and see if the code functions
#Example of code running:
submission_file = "submission.csv" #The file generated in a prior step
best_pairings = find_best_double_helices(submission_file)

print (best_pairings)


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_double_helices_from_pairing(best_pairings, submission_file):
    """Generates 3D plots of the best double helix pairs.

    Args:
      best_pairings: A DataFrame with the best double helix pairs and their similarity scores.
      submission_file: The path to the submission.csv file.
    """
    submission = pd.read_csv(submission_file)  # Load submission data
    submission['structure_num'] = submission['ID'].apply(lambda x: int(x.split('_')[-1])) #Get which point it is

    # Group by ID to obtain coordinates for each helix
    grouped = submission.groupby('ID')

    # Iterate over the best pairings and plot
    for index, row in best_pairings.iterrows():
        helix1_id = row['Helix 1']
        helix2_id = row['Helix 2']
        similarity = row['Similarity']

        #If both values are none, pass
        if (helix1_id == "" and helix2_id == ""):
            continue

        # Extract data for Helix 1
        try:
            group1 = grouped.get_group(helix1_id)  # Get data
            x1 = group1['x_1'].values # Load x cord
            y1 = group1['y_1'].values # Load y cord
            z1 = group1['z_1'].values # Load z cord
        except KeyError:
            print(f"KeyError: Could not find helix ID '{helix1_id}' in submission data.")
            continue

        # Extract data for Helix 2
        try:
            group2 = grouped.get_group(helix2_id) #Load group
            x2 = group2['x_1'].values #Load value x
            y2 = group2['y_1'].values #Load value y
            z2 = group2['z_1'].values #Load value z
        except KeyError:
            print(f"KeyError: Could not find helix ID '{helix2_id}' in submission data.")
            continue

        # Create the Plot:
        fig = plt.figure(figsize=(8, 8))  #Adjust figure size
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(f"Double Helix Pair: {helix1_id} & {helix2_id}")
        try:
            ax.plot(x1, y1, z1, label=f"{helix1_id} (Similarity: {similarity:.2f})")
            ax.plot(x2, y2, z2, label=f"{helix2_id} (Similarity: {similarity:.2f})")

        except Exception as e:
            print(f"Plotting Error: {e}") #Test

        ax.legend()
        plt.show()


#Example values and run. This tests for sample input.
data = {'Helix 1': ['R1107_4', 'R1108_1', 'R1116_2', 'R1117v2_1', 'R1126_2', 'R1128_2', 'R1136_1', 'R1138_1', 'R1149_4', 'R1156_1', 'R1189_3', 'R1190_1'],
        'Helix 2': ['R1107_5', 'R1108_4', 'R1116_4', 'R1117v2_2', 'R1126_4', 'R1128_3', 'R1136_2', 'R1138_5', 'R1149_5', 'R1156_3', 'R1189_5', 'R1190_4'],
        'Similarity': [39.271966, 42.228064, 98.206036, 17.074912, 233.251683, 150.998057, 239.118322, 441.290446, 75.711110, 80.915374, 73.966377, 69.820728]}

best_pairings = pd.DataFrame(data) #Make dataframe
submission_file = "submission.csv" #Load

plot_double_helices_from_pairing(best_pairings, submission_file)


import pandas as pd
import numpy as np
import os

def generate_submission_file(test_file, num_structures=5):
    """
    Generates submission file for the competition, with five predicted structures.

    Args:
      test_file: Path to the test_sequences.csv file.
      num_structures: Number of predicted structures per sequence (default: 5).

    Returns:
      A Pandas DataFrame representing the submission file.
    """

    # Load the test sequences data
    try:
        test_df = pd.read_csv(test_file)
    except FileNotFoundError:
        print(f"Error: Test file '{test_file}' not found.")
        return None

    submission_list = [] # Will store the output

    for num_row_test in range(test_df.shape[0]): #Iterates and create values.
        #Extract the test data and value counts for each value
        row_test = test_df.iloc[num_row_test, :] #Extrat the test data
        target_id = row_test["target_id"] #Map to the correct target id
        sequence = row_test["sequence"] #Sequence will be the structure
        total_pts = len(sequence)    #The total points

        #Iterates through each of the structures and map the file. The task specifies 5 files.
        for number_runs in range(num_structures): #Create all combinations for each target_id

            #Run the results through your algorithm, and store the data points.
            num_points = 3 #Generate random coordinates
            x_coords = np.random.rand(total_pts)  #Replace: Replace coordinate generator
            y_coords = np.random.rand(total_pts)  #Replace: Replace coordinate generator
            z_coords = np.random.rand(total_pts)  #Replace: Replace coordinate generator

            for res_num in range (total_pts): #Iterate to the end of the files

                #Create a unique targetID from the run.
                unique_id = target_id + "_" + str(number_runs + 1) #Each run gets added

                #Get all the data and put the data to data_row
                resname = sequence[res_num] #Get the name and store the value
                data_row = [unique_id, resname, res_num+1, float(x_coords[res_num]), float(y_coords[res_num]), float(z_coords[res_num])]

                #Append data to the submission list
                submission_list.append(data_row)

    #Set header and column names
    header_names = ["ID", "resname", "resid", "x_1", "y_1", "z_1"] #Set each point

    #All values get stored to the dataframe
    df = pd.DataFrame(submission_list, columns=header_names)

    #Export all data into file
    df.to_csv("submission.csv", index = False)

    #Load to test if the file is in the correct format. Can be commented out
    submission = pd.read_csv("submission.csv") #Ensure that datatypes are what is expected
    print (submission.head(50))
    return submission

# Replace this filename to read through the submission algorithm, with test data.
competition_name = "stanford-rna-3d-folding"
filename = f"/kaggle/input/{competition_name}/test_sequences.csv" #Correct filename

submission = generate_submission_file(filename)

#Test for file success or failure to print in the notebook
if (submission is None):
        print ("The algorithm did not pass. Please test with the proper file loaded in the notebook")
else:
        print ("Algorithm has passed. Please submit the submission.csv file.")


import pandas as pd
import math
import os
from IPython.display import display, HTML
import numpy as np
import matplotlib.pyplot as plt  # Import for plotting
from mpl_toolkits.mplot3d import Axes3D  # Import for 3D plotting

# Assume the existence of the following file on the host:
#   - test_sequences.csv: test sequences

def calculate_lref_d0(df):
    """Calculates Lref and d0 for each entry in the DataFrame."""
    df['Lref'] = df['sequence'].apply(len)
    df['d0'] = df['Lref'].apply(lambda lref: 0.6 * math.sqrt(lref - 0.5) - 2.5)
    return df

# Define the base path for the input files in Kaggle
competition_name = "stanford-rna-3d-folding"
base_path = f"/kaggle/input/{competition_name}/"

# Load the CSV files
try:
    validation_csv_path = os.path.join(base_path, "validation_sequences.csv")
    test_csv_path = os.path.join(base_path, "test_sequences.csv")

    # Load the dataframes, raising FileNotFoundError exception if the files can't be found.
    validation_df = pd.read_csv(validation_csv_path)
    test_df = pd.read_csv(test_csv_path)
except FileNotFoundError as e:
    print(f"Error: {e}")
    #If either of the dataframes can't be found, stop execution.
    raise

# Calculate Lref and d0 for both DataFrames
validation_df = calculate_lref_d0(validation_df)
print("<h2>Validation Data:</h2>")
display(HTML(validation_df[['target_id', 'sequence', 'Lref', 'd0']].to_html(index=False)))

test_df = calculate_lref_d0(test_df)
print("<h2>Test Data:</h2>")
display(HTML(test_df[['target_id', 'sequence', 'Lref', 'd0']].to_html(index=False)))

# --- PREDICTION CODE STARTS HERE ---
# REPLACE THIS ENTIRE SECTION WITH YOUR ACTUAL MODEL AND 3D STRUCTURE PREDICTION LOGIC!
# This is placeholder code to generate random coordinates, attempting to simulate
# a helical structure. You MUST adapt this to use your trained model to predict
# the 3D structure of each RNA sequence in 'test_df'.

# Parameters to control the helix generation (adjust these!)
num_structures = 2  # Two helices for a double helix
np.random.seed(42)  # For reproducibility
helix_radius = 5.0  # Radius of the helix
helix_pitch = 3.0  # Vertical distance between turns
z_scaling_factor = 1.0

def generate_helical_structure(sequence_length, structure_number, phase_shift=0):
    """Generates a single helical 3D structure (coordinates for each residue)."""
    angles = np.linspace(0, 4 * np.pi, sequence_length) + phase_shift  # Angles for points along the helix, including phase shift
    random_offsets = np.random.rand(sequence_length) * 1.0  # Add some random noise
    x_coords = helix_radius * np.cos(angles) + random_offsets
    y_coords = helix_radius * np.sin(angles) + random_offsets
    z_coords = helix_pitch * angles  + z_scaling_factor * structure_number
    return x_coords, y_coords, z_coords

all_structure_data = []

# Create a single figure and axes for all plots
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("3D Structures of RNA Sequences")  # General plot title

# Generate the plots together
for index, row in test_df.iterrows():
    target_id = row['target_id']
    sequence = row['sequence']
    sequence_length = len(sequence)

    # Generate a color for the sequence (for plotting) - removed, will use fixed colors for helices.
    # color = np.random.rand(3,)

    # Plot the two helices for each sequence
    x_coords1, y_coords1, z_coords1 = generate_helical_structure(sequence_length, 1)
    x_coords2, y_coords2, z_coords2 = generate_helical_structure(sequence_length, 2, phase_shift=np.pi)  # Add phase shift

    ax.plot(x_coords1, y_coords1, z_coords1, c='red', label=f"{target_id} Helix 1")
    ax.plot(x_coords2, y_coords2, z_coords2, c='blue', label=f"{target_id} Helix 2")

    # Populate structure data for Helix 1
    for residue_index in range(sequence_length):
        resname = sequence[residue_index]
        resid = residue_index + 1
        structure_id = f"{target_id}_1"
        all_structure_data.append([structure_id, resname, resid, x_coords1[residue_index], y_coords1[residue_index], z_coords1[residue_index]])

    # Populate structure data for Helix 2
    for residue_index in range(sequence_length):
        resname = sequence[residue_index]
        resid = residue_index + 1
        structure_id = f"{target_id}_2"
        all_structure_data.append([structure_id, resname, resid, x_coords2[residue_index], y_coords2[residue_index], z_coords2[residue_index]])

ax.legend() # Add a legend
plt.show() # Show plots

#Create the columns based on the structure of all_structure_data
col_names = ['ID', 'resname', 'resid', 'x_1', 'y_1', 'z_1']

# --- PREDICTION CODE ENDS HERE ---

# --- CREATE SUBMISSION FILE ---
submission_df = pd.DataFrame(all_structure_data, columns=col_names)

submission_filename = 'submission.csv'
submission_df.to_csv(submission_filename, index=False)

# --- VERIFY SUBMISSION FILE CREATION ---
if os.path.exists(submission_filename):
    print(f"Submission file '{submission_filename}' created successfully!")
    print("\nFirst 5 rows of submission.csv:")
    display(HTML(submission_df.head().to_html(index=False)))  # Show the first few rows
    print(f"\nSubmission contains: {len(submission_df)} rows")  # Shows the number of rows
else:
    print(f"ERROR: Submission file '{submission_filename}' was NOT created!")

print("Make sure to submit file named submission.csv")


import pandas as pd
import math
import os
from IPython.display import display, HTML
import numpy as np
import matplotlib.pyplot as plt  # Import for plotting
from mpl_toolkits.mplot3d import Axes3D  # Import for 3D plotting
from sklearn.metrics import pairwise_distances

# Assume the existence of the following file on the host:
#   - test_sequences.csv: test sequences

def calculate_lref_d0(df):
    """Calculates Lref and d0 for each entry in the DataFrame."""
    df['Lref'] = df['sequence'].apply(len)
    df['d0'] = df['Lref'].apply(lambda lref: 0.6 * math.sqrt(lref - 0.5) - 2.5)
    return df

# Define the base path for the input files in Kaggle
competition_name = "stanford-rna-3d-folding"
base_path = f"/kaggle/input/{competition_name}/"

# Load the CSV files
try:
    validation_csv_path = os.path.join(base_path, "validation_sequences.csv")
    test_csv_path = os.path.join(base_path, "test_sequences.csv")
    validation_labels_path = os.path.join(base_path, "validation_labels.csv")

    # Load the dataframes, raising FileNotFoundError exception if the files can't be found.
    validation_df = pd.read_csv(validation_csv_path)
    test_df = pd.read_csv(test_csv_path)
    validation_labels_df = pd.read_csv(validation_labels_path)

except FileNotFoundError as e:
    print(f"Error: {e}")
    #If either of the dataframes can't be found, stop execution.
    raise

# Calculate Lref and d0 for both DataFrames
validation_df = calculate_lref_d0(validation_df)
print("<h2>Validation Data:</h2>")
display(HTML(validation_df[['target_id', 'sequence', 'Lref', 'd0']].to_html(index=False)))

test_df = calculate_lref_d0(test_df)
print("<h2>Test Data:</h2>")
display(HTML(test_df[['target_id', 'sequence', 'Lref', 'd0']].to_html(index=False)))

# Extract target IDs from validation_labels.csv
target_ids_from_labels = validation_labels_df['ID'].str.split('_').str[0].unique()

# Find descriptions from test_sequences.csv that match the target IDs
relevant_descriptions = test_df[test_df['target_id'].isin(target_ids_from_labels)][['target_id', 'description']]

# Print the relevant descriptions
print("\n<h2>Relevant Descriptions from test_sequences.csv:</h2>")
display(HTML(relevant_descriptions.to_html(index=False)))

# --- PREDICTION CODE STARTS HERE ---
# REPLACE THIS ENTIRE SECTION WITH YOUR ACTUAL MODEL AND 3D STRUCTURE PREDICTION LOGIC!
# This is placeholder code to generate random coordinates, attempting to simulate
# a helical structure. You MUST adapt this to use your trained model to predict
# the 3D structure of each RNA sequence in 'test_df'.

# Parameters to control the helix generation (adjust these!)
num_structures = 2  # Two helices for a double helix
np.random.seed(42)  # For reproducibility
helix_radius = 5.0  # Radius of the helix
helix_pitch = 3.0  # Vertical distance between turns
z_scaling_factor = 1.0

def generate_helical_structure(sequence_length, structure_number, phase_shift=0):
    """Generates a single helical 3D structure (coordinates for each residue)."""
    angles = np.linspace(0, 4 * np.pi, sequence_length) + phase_shift  # Angles for points along the helix, including phase shift
    random_offsets = np.random.rand(sequence_length) * 1.0  # Add some random noise
    x_coords = helix_radius * np.cos(angles) + random_offsets
    y_coords = helix_radius * np.sin(angles) + random_offsets
    z_coords = helix_pitch * angles  + z_scaling_factor * structure_number
    return x_coords, y_coords, z_coords

all_structure_data = []
results_data = []  # To store results for each sequence

# Create a single figure and axes for all plots
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("3D Structures of RNA Sequences")  # General plot title

# Generate the plots together
for index, row in test_df[test_df['target_id'].isin(target_ids_from_labels)].iterrows(): #Filter test_df
    target_id = row['target_id']
    sequence = row['sequence']
    sequence_length = len(sequence)

    # Generate the two helices
    x_coords1, y_coords1, z_coords1 = generate_helical_structure(sequence_length, 1)
    x_coords2, y_coords2, z_coords2 = generate_helical_structure(sequence_length, 2, phase_shift=np.pi)  # Add phase shift

    ax.plot(x_coords1, y_coords1, z_coords1, c='red', label=f"{target_id} Helix 1")
    ax.plot(x_coords2, y_coords2, z_coords2, c='blue', label=f"{target_id} Helix 2")

    # Combine coordinates for singularity analysis (example: Euclidean distance)
    all_coords = np.column_stack((np.concatenate([x_coords1, x_coords2]),
                                   np.concatenate([y_coords1, y_coords2]),
                                   np.concatenate([z_coords1, z_coords2])))

    # Calculate distances.  Experiment with different distance metrics.
    distance_matrix = pairwise_distances(all_coords)

    # Calculate average distance (a simple measure of "singularity")
    avg_distance = np.mean(distance_matrix)

    results_data.append([target_id, sequence_length, avg_distance])

    # Populate structure data (unchanged)
    for residue_index in range(sequence_length):
        resname = sequence[residue_index]
        resid = residue_index + 1
        structure_id = f"{target_id}_1"
        all_structure_data.append([structure_id, resname, resid, x_coords1[residue_index], y_coords1[residue_index], z_coords1[residue_index]])
    for residue_index in range(sequence_length):
        resname = sequence[residue_index]
        resid = residue_index + 1
        structure_id = f"{target_id}_2"
        all_structure_data.append([structure_id, resname, resid, x_coords2[residue_index], y_coords2[residue_index], z_coords2[residue_index]])

ax.legend() # Add a legend
plt.show() # Show plots

# Create results DataFrame and display table
results_df = pd.DataFrame(results_data, columns=['target_id', 'sequence_length', 'average_distance'])
print("<h2>Singularity Analysis Results:</h2>")
display(HTML(results_df.to_html(index=False)))

# Create singularity comparison plot
plt.figure(figsize=(10, 6))
plt.bar(results_df['target_id'], results_df['average_distance'])
plt.xlabel("Target ID")
plt.ylabel("Average Distance (Singularity Measure)")
plt.title("Comparison of Singularity Measures")
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for readability
plt.tight_layout()
plt.show()

#Create the columns based on the structure of all_structure_data
col_names = ['ID', 'resname', 'resid', 'x_1', 'y_1', 'z_1']

# --- CREATE SUBMISSION FILE ---
submission_df = pd.DataFrame(all_structure_data, columns=col_names)

submission_filename = 'submission.csv'
submission_df.to_csv(submission_filename, index=False)

# --- VERIFY SUBMISSION FILE CREATION ---
if os.path.exists(submission_filename):
    print(f"Submission file '{submission_filename}' created successfully!")
    print("\nFirst 5 rows of submission.csv:")
    display(HTML(submission_df.head().to_html(index=False)))  # Show the first few rows
    print(f"\nSubmission contains: {len(submission_df)} rows")  # Shows the number of rows
else:
    print(f"ERROR: Submission file '{submission_filename}' was NOT created!")

print("Make sure to submit file named submission.csv")


import pandas as pd
import math
import os
from IPython.display import display, HTML
import numpy as np
# import matplotlib.pyplot as plt  # No need for matplotlib in submission
# from mpl_toolkits.mplot3d import Axes3D  # No need for 3D plotting in submission
# from sklearn.metrics import pairwise_distances  # No need for singularity analysis

# Assume the existence of the following file on the host:
#   - test_sequences.csv: test sequences

def calculate_lref_d0(df):
    """Calculates Lref and d0 for each entry in the DataFrame."""
    df['Lref'] = df['sequence'].apply(len)
    df['d0'] = df['Lref'].apply(lambda lref: 0.6 * math.sqrt(lref - 0.5) - 2.5)
    return df

# Define the base path for the input files in Kaggle
competition_name = "stanford-rna-3d-folding"
base_path = f"/kaggle/input/{competition_name}/"

# Load the CSV files
try:
    # validation_csv_path = os.path.join(base_path, "validation_sequences.csv") #No need to load validation data in submission
    test_csv_path = os.path.join(base_path, "test_sequences.csv")
    # validation_labels_path = os.path.join(base_path, "validation_labels.csv") #No need to load validation labels in submission

    # Load the dataframes, raising FileNotFoundError exception if the files can't be found.
    # validation_df = pd.read_csv(validation_csv_path) #No need to load validation data in submission
    test_df = pd.read_csv(test_csv_path)
    # validation_labels_df = pd.read_csv(validation_labels_path) #No need to load validation labels in submission

except FileNotFoundError as e:
    print(f"Error: {e}")
    #If either of the dataframes can't be found, stop execution.
    raise

# Calculate Lref and d0 for the test DataFrame
test_df = calculate_lref_d0(test_df)
# print("<h2>Test Data:</h2>")
# display(HTML(test_df[['target_id', 'sequence', 'Lref', 'd0']].to_html(index=False))) #No need to display test data in submission

# Extract target IDs from test_sequences.csv
target_ids = test_df['target_id'].unique()

# --- PREDICTION CODE STARTS HERE ---
# REPLACE THIS ENTIRE SECTION WITH YOUR ACTUAL MODEL AND 3D STRUCTURE PREDICTION LOGIC!
# This is placeholder code to generate random coordinates, attempting to simulate
# a helical structure. You MUST adapt this to use your trained model to predict
# the 3D structure of each RNA sequence in 'test_df'.

# Parameters to control the helix generation (adjust these!)
num_structures = 5  # Predict five structures
np.random.seed(42)  # For reproducibility
helix_radius = 5.0  # Radius of the helix
helix_pitch = 3.0  # Vertical distance between turns

def generate_helical_structure(sequence_length, structure_number, phase_shift=0):
    """Generates a single helical 3D structure (coordinates for each residue)."""
    angles = np.linspace(0, 4 * np.pi, sequence_length) + phase_shift  # Angles for points along the helix, including phase shift
    random_offsets = np.random.rand(sequence_length) * 1.0  # Add some random noise
    x_coords = helix_radius * np.cos(angles) + random_offsets
    y_coords = helix_radius * np.sin(angles) + random_offsets
    z_coords = helix_pitch * angles  + structure_number #Linear progression for z coords
    return x_coords, y_coords, z_coords

all_structure_data = []

for target_id in target_ids:  # Iterate through all target IDs in test_df
    row = test_df[test_df['target_id'] == target_id].iloc[0]
    sequence = row['sequence']
    sequence_length = len(sequence)

    # Generate multiple structures (5 in this case)
    for structure_index in range(1, num_structures + 1):
        x_coords, y_coords, z_coords = generate_helical_structure(sequence_length, structure_index)

        for residue_index in range(sequence_length):
            resname = sequence[residue_index]
            resid = residue_index + 1
            all_structure_data.append([f"{target_id}_{resid}", resname, resid, x_coords[residue_index], y_coords[residue_index], z_coords[residue_index]])

# --- CREATE SUBMISSION FILE ---
#Create the columns based on the structure of all_structure_data
col_names = ['ID', 'resname', 'resid', 'x_1', 'y_1', 'z_1']
submission_df = pd.DataFrame(all_structure_data, columns=col_names)

# Create new dataframe with the required submission format
reshaped_data = []

for index, row in submission_df.iterrows():
    ID, resname, resid, x_1, y_1, z_1 = row

    # Extract base ID, resname, and resid
    base_id = ID.rsplit('_', 1)[0]  # Get target_id from ID
    resname = row['resname']
    resid = row['resid']

    # Create a dictionary to store the reshaped data
    reshaped_row = {'ID': f"{base_id}_{resid}", 'resname': resname, 'resid': resid}

    # Find all rows with the same base ID and resid
    matching_rows = submission_df[(submission_df['ID'].str.startswith(base_id)) & (submission_df['resid'] == resid)]

    # Iterate through the matching rows and add the coordinates to the reshaped row
    for i in range(1, num_structures + 1):
        coords = matching_rows[matching_rows['ID'].str.endswith(f"_{resid}")][['x_1', 'y_1', 'z_1']].values.flatten()
        if len(coords) > 0:
            reshaped_row[f'x_{i}'] = coords[0]
            reshaped_row[f'y_{i}'] = coords[1]
            reshaped_row[f'z_{i}'] = coords[2]

    reshaped_data.append(reshaped_row)

reshaped_df = pd.DataFrame(reshaped_data)

submission_filename = 'submission.csv'
reshaped_df.to_csv(submission_filename, index=False)

# --- VERIFY SUBMISSION FILE CREATION ---
if os.path.exists(submission_filename):
    print(f"Submission file '{submission_filename}' created successfully!")
    print("\nFirst 5 rows of submission.csv:")
    display(HTML(reshaped_df.head().to_html(index=False)))  # Show the first few rows
    print(f"\nSubmission contains: {len(reshaped_df)} rows")  # Shows the number of rows
else:
    print(f"ERROR: Submission file '{submission_filename}' was NOT created!")

print("Make sure to submit file named submission.csv")

