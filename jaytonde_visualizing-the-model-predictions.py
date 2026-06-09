import pandas as pd
import numpy as np
from tqdm import tqdm
import plotly.graph_objects as go
from plotly.subplots import make_subplots


root_path                = '/kaggle/input/stanford-rna-3d-folding'
experiment_sub_file_path = "/kaggle/input/ribonanzanet-3d-inference/submission.csv"


valid_sequences        = pd.read_csv(f"{root_path}/validation_sequences.csv")
valid_labels           = pd.read_csv(f"{root_path}/validation_labels.csv")
valid_labels["pdb_id"] = valid_labels["ID"].apply(lambda x: x.split("_")[0])


preds_df           = pd.read_csv(experiment_sub_file_path)
preds_df["pdb_id"] = preds_df["ID"].apply(lambda x: x.split("_")[0])


all_xyz=[]
all_preds = []

for pdb_id in tqdm(valid_sequences['target_id']):
    df = valid_labels[valid_labels["pdb_id"]==pdb_id]
    xyz=df[['x_1','y_1','z_1']].to_numpy().astype('float32')
    xyz[xyz<-1e17]=float('Nan');
    all_xyz.append(xyz)

    temp_arr = []
    sub_preds_df = preds_df[preds_df['pdb_id']==pdb_id]
    
    xyz_preds1=sub_preds_df[['x_1','y_1','z_1']].to_numpy().astype('float32')
    xyz_preds1[xyz_preds1<-1e17]=float('Nan');
    temp_arr.append(xyz_preds1)

    xyz_preds2=sub_preds_df[['x_2','y_2','z_2']].to_numpy().astype('float32')
    xyz_preds2[xyz_preds2<-1e17]=float('Nan');
    temp_arr.append(xyz_preds2)

    xyz_preds3=sub_preds_df[['x_3','y_3','z_3']].to_numpy().astype('float32')
    xyz_preds3[xyz_preds3<-1e17]=float('Nan');
    temp_arr.append(xyz_preds3)

    xyz_preds4=sub_preds_df[['x_4','y_4','z_4']].to_numpy().astype('float32')
    xyz_preds4[xyz_preds4<-1e17]=float('Nan');
    temp_arr.append(xyz_preds4)

    xyz_preds5=sub_preds_df[['x_5','y_5','z_5']].to_numpy().astype('float32')
    xyz_preds5[xyz_preds5<-1e17]=float('Nan');
    temp_arr.append(xyz_preds5)
    
    all_preds.append(temp_arr)

len(all_xyz), len(all_preds)


#pack data into a dictionary

data={
      "sequence":valid_sequences['sequence'].to_list(),
      "xyz": all_xyz,
      "preds" : all_preds
}


# Access the target array
target = data['xyz'][2]
predictions  = data['preds'][2]

# Number of predictions
num_preds = len(predictions)


import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Code to plot only the target structure
if isinstance(target, np.ndarray) and target.ndim == 2 and target.shape[1] == 3:
    x_t, y_t, z_t = target[:, 0], target[:, 1], target[:, 2]
    N_t = len(target)
    sequence_indices_t = np.arange(N_t)

    # Create a figure with a single subplot for the target
    fig = make_subplots(
        rows=1,
        cols=1, # Only one column for the target plot
        specs=[[{'type': 'scatter3d'}]], # Specify 3D plot type
        subplot_titles=['Target'] # Title for the single subplot
    )

    # Add the 3D scatter plot trace for the target structure
    fig.add_trace(go.Scatter3d(
        x=x_t, y=y_t, z=z_t,
        mode='lines+markers', # Show backbone line and markers
        line=dict(color='red', width=2), # Style the backbone line (e.g., red)
        marker=dict(
            size=3, # Size of the markers
            color=sequence_indices_t, # Color markers sequentially
            colorscale='Viridis',   # Colorscale for markers
            opacity=0.8,
            colorbar=dict(title='Residue Index') # Add the colorbar
        ),
        text=[f'Residue {j}' for j in sequence_indices_t], # Hover text
        hoverinfo='text+x+y+z' # Information to display on hover
    ), row=1, col=1) # Assign trace to the first (and only) subplot cell

    # Update the layout for the single plot
    fig.update_layout(
        height=400,  # Adjust height as needed
        width=600, # Adjust width for a single plot
        title_text='Target RNA Structure', # Title for the figure
        showlegend=False, # Hide legend as colorbar is sufficient
        margin=dict(l=10, r=10, b=10, t=50) # Adjust margins
    )

    # Display the figure
    fig.show(renderer='iframe') # Use 'iframe' or your preferred renderer

else:
    # Print a warning if the target data is not in the expected format
    print(f"Warning: Target data is not a valid Nx3 numpy array. Cannot plot.")


if num_preds != 5:
    print(f"Error: Expected 5 predictions in preds[2], but found {num_preds}.")
else:
    fig = make_subplots(
        rows=1,
        cols=num_preds, # Add one column for the target
        specs=[[{'type': 'scatter3d'}] * (num_preds)], # Specify 3D plot type for each subplot
        subplot_titles=[f'Prediction {i}' for i in range(num_preds)] # Titles including 'Target'
    )

    # Iterate through each prediction (xyz coordinate array) and add it to the figure
    for i, xyz in enumerate(predictions):
        # Ensure xyz is a numpy array
        if not isinstance(xyz, np.ndarray) or xyz.ndim != 2 or xyz.shape[1] != 3:
             print(f"Warning: Prediction {i+1} is not a valid Nx3 numpy array. Skipping.")
             continue

        x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        N = len(xyz)
        sequence_indices = np.arange(N)

        # Add the 3D scatter plot trace for the current prediction
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode='lines+markers', # Show backbone line and markers for each residue
            line=dict(color='grey', width=2), # Style the backbone line
            marker=dict(
                size=3, # Size of the markers
                color=sequence_indices, # Color markers sequentially
                colorscale='Viridis',   # Colorscale for markers
                opacity=0.8,
                # Remove conditional colorbar here, will add one to the target plot
                # colorbar=dict(title='Residue Index') if i == num_preds - 1 else None
            ),
            text=[f'Residue {j}' for j in sequence_indices], # Hover text
            hoverinfo='text+x+y+z' # Information to display on hover
        ), row=1, col=i + 1) # Assign trace to the correct subplot cell (cols 1 to 5)

    fig.update_layout(
        height=400,  # Adjust height as needed
        width=1800, # Increase width to accommodate 6 plots (approx 300px per plot)
        title_text='Comparison of 5 RNA Structure Predictions and Target Structure', # Updated title
        showlegend=False, # Hide legend as colors are self-explanatory with colorbar
        margin=dict(l=10, r=10, b=10, t=50) # Adjust margins
    )

    fig.show(renderer='iframe') # Use 'iframe' or your preferred renderer




