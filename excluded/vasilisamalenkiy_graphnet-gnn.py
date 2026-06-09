# Import necessary libraries.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load data
geometry = pd.read_csv('/kaggle/input/sensor-geometry-csv/sensor_geometry.csv')
meta = pd.read_parquet('/kaggle/input/train-meta-parquet/train_meta.parquet')
batch = pd.read_parquet('/kaggle/input/batch-1-parquet/batch_1.parquet')

# Select a unique event ID
unique_event_id = batch.index.unique()[30]

# Get metadata and pulse data
event_meta = meta[meta['event_id'] == unique_event_id]
event_data = batch.loc[batch.index == unique_event_id].reset_index()

# Merge sensor geometry
event_data = event_data.merge(geometry, on='sensor_id', how='left')

print(event_data.head())

# Split into main and auxiliary pulses
main_pulses = event_data[event_data['auxiliary'] == 0]
aux_pulses = event_data[event_data['auxiliary'] == 1]

# Neutrino direction (unit vector)
azimuth = event_meta['azimuth'].values[0]
zenith = event_meta['zenith'].values[0]
r_true = np.array([
    np.cos(azimuth) * np.sin(zenith),
    np.sin(azimuth) * np.sin(zenith),
    np.cos(zenith)
])

# Plot
fig = plt.figure(figsize=(16, 8))
vmin = event_data['time'].min()
vmax = event_data['time'].max()

# Left plot: auxiliary == False
ax1 = fig.add_subplot(121, projection='3d')
sc1 = ax1.scatter(main_pulses['x'], main_pulses['y'], main_pulses['z'],
                  c=main_pulses['time'], s=main_pulses['charge'] * 40,
                  cmap='gist_rainbow', vmin=vmin, vmax=vmax, alpha=0.8)
ax1.set_title('auxiliary == False')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('z')

# Center and scaled direction vector
center1 = np.array([main_pulses['x'].mean(),
                    main_pulses['y'].mean(),
                    main_pulses['z'].mean()])
range1 = np.ptp(main_pulses[['x', 'y', 'z']].values, axis=0)  # peak-to-peak range per axis
scale1 = 0.5 * np.linalg.norm(range1)  # half the diagonal of the bounding box
ax1.quiver(center1[0], center1[1], center1[2],
           r_true[0], r_true[1], r_true[2],
           color='r', length=scale1, linewidth=2)

# Right plot: auxiliary == True
ax2 = fig.add_subplot(122, projection='3d')
sc2 = ax2.scatter(aux_pulses['x'], aux_pulses['y'], aux_pulses['z'],
                  c=aux_pulses['time'], s=aux_pulses['charge'] * 40,
                  cmap='gist_rainbow', vmin=vmin, vmax=vmax, alpha=0.8)
ax2.set_title('auxiliary == True')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_zlabel('z')

# Center and scaled direction vector
center2 = np.array([aux_pulses['x'].mean(),
                    aux_pulses['y'].mean(),
                    aux_pulses['z'].mean()])
range2 = np.ptp(aux_pulses[['x', 'y', 'z']].values, axis=0)
scale2 = 0.5 * np.linalg.norm(range2)
ax2.quiver(center2[0], center2[1], center2[2],
           r_true[0], r_true[1], r_true[2],
           color='r', length=scale2, linewidth=2)

# Shared colorbar
cbar_ax = fig.add_axes([0.49, 0.15, 0.015, 0.7])  # [left, bottom, width, height]
cbar = fig.colorbar(sc2, cax=cbar_ax)
cbar.set_label('time')

# Main title
plt.suptitle(f'Event: {unique_event_id}\n(azimuth = {azimuth:.2f} rad, zenith = {zenith:.2f} rad)', fontsize=12)
plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.05, wspace=0.3)
plt.show()


%%bash
pip install --upgrade --user ipywidgets==8.0.5
jupyter nbextension install --user --py widgetsnbextension
jupyter nbextension enable  --user --py widgetsnbextension


import ipywidgets as widgets
print("Python ipywidgets:", widgets.__version__)



from IPython.display import display
slider = widgets.IntSlider(description="Test", min=0, max=50, value=25)
display(slider)


from IPython.display import display
import ipywidgets as widgets
import time

# Create an IntProgress bar
progress = widgets.IntProgress(
    value=0,
    min=0,
    max=100,
    step=1,
    description='Loading:',
    bar_style='',        # 'info', 'success', 'warning', 'danger' or ''
    orientation='horizontal'
)
display(progress)

# Simulate work
for i in range(101):
    time.sleep(0.05)
    progress.value = i



# Moving software to working disk
!rm  -r software
!scp -r /kaggle/input/graphnet-and-dependencies/software .

# Installing dependencies
!pip install /kaggle/working/software/dependencies/torch-1.11.0+cu115-cp37-cp37m-linux_x86_64.whl
!pip install /kaggle/working/software/dependencies/torch_cluster-1.6.0-cp37-cp37m-linux_x86_64.whl
!pip install /kaggle/working/software/dependencies/torch_scatter-2.0.9-cp37-cp37m-linux_x86_64.whl
!pip install /kaggle/working/software/dependencies/torch_sparse-0.6.13-cp37-cp37m-linux_x86_64.whl
!pip install /kaggle/working/software/dependencies/torch_geometric-2.0.4.tar.gz

# Installing GraphNeT
!cd software/graphnet;pip install --no-index --find-links="/kaggle/working/software/dependencies" -e .[torch]

# Appending to PATH
import sys
sys.path.append('/kaggle/working/software/graphnet/src')

import graphnet


import graphnet


# Importing packages required for conversion
import pyarrow.parquet as pq
import sqlite3
import pandas as pd
import sqlalchemy
from tqdm import tqdm
import os
from typing import Any, Dict, List, Optional
import numpy as np

from graphnet.data.sqlite.sqlite_utilities import create_table

def load_input(meta_batch: pd.DataFrame, input_data_folder: str) -> pd.DataFrame:
    """
    Loads the corresponding detector readings associated with the metadata batch.

    Args:
        meta_batch (pd.DataFrame): Metadata containing batch information, specifically 'batch_id'.
        input_data_folder (str): Folder path containing the parquet input files.

    Returns:
        pd.DataFrame: A DataFrame containing detector readings merged with sensor positions.

    Raises:
        AssertionError: If there are multiple 'batch_id' values in the 'meta_batch' DataFrame.
    """
    batch_id = pd.unique(meta_batch['batch_id'])

    # Ensure that the metadata batch contains only one batch_id
    assert len(batch_id) == 1, "contains multiple batch_ids. Did you set the batch_size correctly?"
    
    # Load the corresponding detector readings from a parquet file
    detector_readings = pd.read_parquet(path=f'{input_data_folder}/batch_{batch_id[0]}.parquet')

    # Retrieve the sensor positions based on sensor_id
    sensor_positions = geometry_table.loc[detector_readings['sensor_id'], ['x', 'y', 'z']]
    sensor_positions.index = detector_readings.index

    # Add the sensor position columns to the detector readings
    for column in sensor_positions.columns:
        if column not in detector_readings.columns:
            detector_readings[column] = sensor_positions[column]

    # Convert auxiliary column from boolean to integer (1/0)
    detector_readings['auxiliary'] = detector_readings['auxiliary'].replace({True: 1, False: 0})

    # Return the modified DataFrame with reset index
    return detector_readings.reset_index()

def add_to_table(database_path: str,
                 df: pd.DataFrame,
                 table_name: str,
                 is_primary_key: bool) -> None:
    """
    Writes metadata to an SQLite table.

    Args:
        database_path (str): Path to the SQLite database file.
        df (pd.DataFrame): DataFrame to be written to the SQLite table.
        table_name (str): Name of the table in the SQLite database.
        is_primary_key (bool): True if the 'event_id' column is a primary key; otherwise False.
    
    Raises:
        sqlite3.OperationalError: If there is an error during table creation or data insertion.
    """
    try:
        # Attempt to create the table, if it doesn't already exist
        create_table(columns=df.columns,
                     database_path=database_path,
                     table_name=table_name,
                     integer_primary_key=is_primary_key,
                     index_column='event_id')
    except sqlite3.OperationalError as e:
        # If the table already exists, ignore the error
        if 'already exists' not in str(e):
            raise e

    # Create a connection to the SQLite database using SQLAlchemy
    engine = sqlalchemy.create_engine("sqlite:///" + database_path)
    
    # Write the DataFrame to the table (appending the data)
    df.to_sql(table_name, con=engine, index=False, if_exists="append", chunksize=200000)
    
    # Close the engine connection
    engine.dispose()
    return

def convert_to_sqlite(meta_data_path: str,
                      database_path: str,
                      input_data_folder: str,
                      batch_size: int = 200000,
                      batch_ids: Optional[List[int]] = None) -> None:
    """
    Converts a selection of the Competition's Parquet files to a single SQLite database.

    Args:
        meta_data_path (str): Path to the metadata Parquet file.
        database_path (str): Path to the SQLite database file (e.g., '/my_folder/data/my_new_database.db').
        input_data_folder (str): Folder containing the Parquet input files.
        batch_size (int): Number of rows to extract from the metadata file at a time. Default is 200000.
        batch_ids (List[int], optional): List of specific batch IDs to be converted. Defaults to None, 
                                          which will convert all batches from 1 to 660.

    Raises:
        AssertionError: If 'batch_ids' is provided and is not a list.
    """
    if batch_ids is None:
        batch_ids = np.arange(1, 661, 1).tolist()
    else:
        assert isinstance(batch_ids, list), "Variable 'batch_ids' must be a list."

    # Ensure the database path ends with '.db'
    if not database_path.endswith('.db'):
        database_path = database_path + '.db'

    # Initialize the iterator to read the parquet file in batches
    meta_data_iter = pq.ParquetFile(meta_data_path).iter_batches(batch_size=batch_size)

    batch_id = 1
    converted_batches = []
    progress_bar = tqdm(total=len(batch_ids))  # Progress bar to track batch conversion

    # Process each batch from the parquet file
    for meta_data_batch in meta_data_iter:
        if batch_id in batch_ids:
            # Convert the batch into a pandas DataFrame
            meta_data_batch = meta_data_batch.to_pandas()

            # Write the metadata batch to the SQLite database
            add_to_table(database_path=database_path,
                         df=meta_data_batch,
                         table_name='meta_table',
                         is_primary_key=True)

            # Load corresponding pulses data using the metadata
            pulses = load_input(meta_batch=meta_data_batch, input_data_folder=input_data_folder)
            
            # Free memory by deleting the metadata batch
            del meta_data_batch

            # Write the pulses data to the SQLite database
            add_to_table(database_path=database_path,
                         df=pulses,
                         table_name='pulse_table',
                         is_primary_key=False)
            
            # Free memory by deleting the pulses data
            del pulses

            # Update the progress bar
            progress_bar.update(1)
            converted_batches.append(batch_id)

        batch_id += 1

        # If all specified batch IDs are processed, stop the loop
        if len(batch_ids) == len(converted_batches):
            break

    # Close the progress bar
    progress_bar.close()

    # Free memory by deleting the iterator
    del meta_data_iter

    # Print completion message
    print(f'Conversion Complete! Database available at\n {database_path}')


!cp /kaggle/input/batch-1/batch_1.db .
!cp /kaggle/input/batch-51/batch_51.db .


from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import sqlite3
from tqdm import tqdm

def make_selection(df: pd.DataFrame, pulse_threshold: int = 200, test_size: float = 0.20, random_state: int = 42) -> None:
    """
    Creates training and validation selections from the provided dataframe.
    
    This function splits the events into training and validation sets, 
    ensuring that the events in both sets satisfy the condition of having
    n_pulses <= pulse_threshold by default. The function allows customization 
    of the test size and random seed for reproducibility.

    Args:
    - df (pd.DataFrame): The dataframe containing event data with pulse information.
    - pulse_threshold (int): The maximum number of pulses allowed in an event to be included. Default is 200.
    - test_size (float): Fraction of data to be used for validation. Default is 0.20.
    - random_state (int): Random seed for reproducibility of the split. Default is 42.

    Returns:
    - None: The function saves the selected training and validation datasets as CSV files.
    """
    # Generate event indices for splitting
    n_events = np.arange(0, len(df), 1)
    
    # Split data into training and validation sets (80-20 by default)
    train_selection, validate_selection = train_test_split(n_events, 
                                                           shuffle=True, 
                                                           random_state=random_state, 
                                                           test_size=test_size)

    # Initialize the train and validate columns
    df['train'] = 0
    df['validate'] = 0
    
    # Assign selections
    df['train'][train_selection] = 1
    df['validate'][validate_selection] = 1
    
    # Ensure correct sizes
    assert len(train_selection) == sum(df['train'])
    assert len(validate_selection) == sum(df['validate'])

    # Remove events with pulse counts exceeding the threshold
    df['train'][df['n_pulses'] > pulse_threshold] = 0
    df['validate'][df['n_pulses'] > pulse_threshold] = 0
    
    # Save the filtered selections to CSV files
    for selection in ['train', 'validate']:
        df.loc[df[selection] == 1, :].to_csv(f'{selection}_selection_max_{pulse_threshold}_pulses.csv')
    
    return

def get_number_of_pulses(db: str, event_id: int, pulsemap: str) -> int:
    """
    Retrieves the number of pulses for a given event from the database.
    
    Args:
    - db (str): Path to the SQLite database.
    - event_id (int): The unique identifier of the event.
    - pulsemap (str): The name of the pulse table in the database.

    Returns:
    - int: The number of pulses associated with the event.
    """
    with sqlite3.connect(db) as con:
        query = f'SELECT event_id FROM {pulsemap} WHERE event_id = {event_id} LIMIT 20000'
        data = con.execute(query).fetchall()
    return len(data)

def count_pulses(database: str, pulsemap: str) -> pd.DataFrame:
    """
    Counts the number of pulses for each event in the database.
    
    This function queries the database, retrieves the event IDs, and 
    calculates the pulse count for each event. The resulting data is 
    stored in a dataframe and saved as a CSV file.

    Args:
    - database (str): Path to the SQLite database.
    - pulsemap (str): The name of the pulse table in the database.

    Returns:
    - pd.DataFrame: A dataframe containing event IDs and their corresponding pulse counts.
    """
    # Retrieve all event IDs from the database
    with sqlite3.connect(database) as con:
        query = 'SELECT event_id FROM meta_table'
        events = pd.read_sql(query, con)
    
    # Initialize count storage
    counts = {'event_id': [], 'n_pulses': []}
    
    # Count pulses for each event
    for event_id in tqdm(events['event_id']):
        pulse_count = get_number_of_pulses(database, event_id, pulsemap)
        counts['event_id'].append(event_id)
        counts['n_pulses'].append(pulse_count)
    
    # Create a dataframe and save it
    df = pd.DataFrame(counts)
    df.to_csv('counts.csv', index=False)
    
    return df


# Training - train the model on known data
# Validation - validating model w/ known data
# Testing - applying model to unknown/new data
# Define pulsemap and database file paths
pulsemap = 'pulse_table' # Contains time-series data of light pulses detected by Digital Optical Modules (DOMs) in the ice. 
database = '/kaggle/working/batch_1.db' # Path to SQLite database containing event IDs

# Count pulses for each event ID in the database 
df = count_pulses(database, pulsemap)
# Make training and validation selections cutting all events with pulse counts above the threshold. 
make_selection(df = df, pulse_threshold =  200)


import matplotlib.pyplot as plt
import pandas as pd

# Plot the number of pulses detected for each event ID as a function of the number of events for that given number of pulse counts (energy).
fig = plt.figure(figsize=(6,4), constrained_layout = True)
plt.hist(df['n_pulses'], histtype = 'step', label = 'batch_1', bins = np.arange(0,400,1))
plt.xlabel('# of Pulses', size = 15);
plt.xticks(size = 15);
plt.yticks(size = 15);
plt.plot(np.repeat(200,2), [0, 4000], label = f'Selection\n{np.round((sum(df["n_pulses"]<= 200)/len(df))*100, 1)} % pass' ) 
plt.legend(frameon = False, fontsize = 15);


print(f'Event with highest number of pulses counted: {df["n_pulses"].max()}')


# This is the BULK of the DynEdge GNN and what I will be experimenting with in addition to some data preprocessing steps to 
# minimize the angle between the true and reconstructed neutrino direction vector. 

# Importing required libraries and modules
from pytorch_lightning.callbacks import EarlyStopping  # Stops training when validation loss stops improving
from torch.optim.adam import Adam  # Optimizer - adapts learning rate per parameter to minimize the loss
from graphnet.data.constants import FEATURES, TRUTH  # Predefined sets of features and ground truth labels
from graphnet.models import StandardModel  # Wrapper that combines the detector, GNN, and reconstruction task
from graphnet.models.detector.icecube import IceCubeKaggle  # Detector abstraction for IceCube Kaggle dataset
from graphnet.models.gnn import DynEdge  # The main Graph Neural Network model used here
from graphnet.models.graph_builders import KNNGraphBuilder  # Builds event-wise graphs based on spatial proximity
from graphnet.models.task.reconstruction import (
    DirectionReconstructionWithKappa,  # 3D direction reconstruction task with uncertainty (kappa)
    ZenithReconstructionWithKappa,
    AzimuthReconstructionWithKappa,
)
from graphnet.training.callbacks import ProgressBar, PiecewiseLinearLR  # Visual feedback and LR scheduler
from graphnet.training.loss_functions import VonMisesFisher3DLoss, VonMisesFisher2DLoss  # Loss functions for directional data
from graphnet.training.labels import Direction  # Wrapper for retrieving 3D direction labels
from graphnet.training.utils import make_dataloader  # Utility to create PyTorch dataloaders from IceCube DB
from graphnet.utilities.logging import get_logger  # Logging
from pytorch_lightning import Trainer  # Training orchestration
import pandas as pd  # Data processing
import os  # File and directory handling
from typing import Dict, Any, List  # Type hints

# Initialize logger
logger = get_logger()

# KEY FUNCTION I'M (ADJUSTING)
# Build and return a StandardModel configured to perform direction reconstruction using DynEdge GNN
def build_model(config: Dict[str,Any], train_dataloader: Any) -> StandardModel:
    """Builds a DynEdge-based GNN for directional reconstruction on IceCube data"""

    # Step 1: Define the detector model and graph construction method (KNN)
    detector = IceCubeKaggle(
        graph_builder=KNNGraphBuilder(nb_nearest_neighbours=12),  # Build graph with k=8 nearest hits
    )

    # Step 2: Define the GNN model (DynEdge) with input from detector and global pooling
    gnn = DynEdge(
        nb_inputs=detector.nb_outputs,  # Input dim = number of output features from detector
        global_pooling_schemes=["min", "max", "mean"],  # Use multiple pooling types across the graph
    )

    # Step 3: Define the task (what the GNN will predict) — direction vector + uncertainty (kappa)
    if config["target"] == 'direction':
        task = DirectionReconstructionWithKappa(
            hidden_size=gnn.nb_outputs,  # Hidden layer size matches GNN output
            target_labels=config["target"],
            loss_function=VonMisesFisher3DLoss(),  # Angular loss for 3D directions THIS IS WHERE LOSS FUNCTION IS SPECIFIED (ADJUST)
        )
        prediction_columns = [config["target"] + "_x", 
                              config["target"] + "_y", 
                              config["target"] + "_z", 
                              config["target"] + "_kappa"]  # x/y/z/kappa outputs
        additional_attributes = ['zenith', 'azimuth', 'event_id']  # Add human-readable angles and IDs

    # Step 4: Combine all components into a trainable model
    model = StandardModel(
        detector=detector,
        gnn=gnn,
        tasks=[task],
        optimizer_class=Adam,  # Use Adam optimizer
        optimizer_kwargs={"lr": 1e-03, "eps": 1e-03},  # Learning rate and numerical stability term COULD ALSO ADJUST (ADJUST)
        scheduler_class=PiecewiseLinearLR,  # Learning rate schedule: ramp up then decay (ADJUST)
        scheduler_kwargs={
            "milestones": [
                0,
                len(train_dataloader) / 2,
                len(train_dataloader) * config["fit"]["max_epochs"],
            ],
            "factors": [1e-02, 1, 1e-02],  # Initial LR factor, plateau, then decay (ADJUST - data quality > rate of learning)
        },
        scheduler_config={"interval": "step"},  # Update LR per step, not epoch (CLAIRFY)
    )
    model.prediction_columns = prediction_columns
    model.additional_attributes = additional_attributes
    
    return model

# Load a pretrained DynEdge model from a saved state dictionary
def load_pretrained_model(config: Dict[str,Any], state_dict_path: str = '/kaggle/input/dynedge-pretrained/dynedge_pretrained_batch_1_to_50/state_dict.pth') -> StandardModel:
    train_dataloader, _ = make_dataloaders(config)
    model = build_model(config, train_dataloader)
    model.load_state_dict(state_dict_path)  # Load model weights from file
    model.prediction_columns = [config["target"] + "_x", 
                                config["target"] + "_y", 
                                config["target"] + "_z", 
                                config["target"] + "_kappa"]
    model.additional_attributes = ['zenith', 'azimuth', 'event_id']
    return model

# Create PyTorch dataloaders for training and validation
def make_dataloaders(config: Dict[str, Any]) -> List[Any]:
    """Constructs training and validation dataloaders for training with early stopping."""
    
    # Training dataloader: shuffles data, uses subset of events from CSV
    train_dataloader = make_dataloader(
        db=config['path'],
        selection=pd.read_csv(config['train_selection'])[config['index_column']].ravel().tolist(),
        pulsemaps=config['pulsemap'],
        features=features,
        truth=truth,
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        shuffle=True,
        labels={'direction': Direction()},
        index_column=config['index_column'],
        truth_table=config['truth_table'],
    )

    # Validation dataloader: no shuffle, used for monitoring performance
    validate_dataloader = make_dataloader(
        db=config['path'],
        selection=pd.read_csv(config['validate_selection'])[config['index_column']].ravel().tolist(),
        pulsemaps=config['pulsemap'],
        features=features,
        truth=truth,
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        shuffle=False,
        labels={'direction': Direction()},
        index_column=config['index_column'],
        truth_table=config['truth_table'],
    )
    return train_dataloader, validate_dataloader

# Train DynEdge model from scratch with early stopping and learning rate scheduling
def train_dynedge_from_scratch(config: Dict[str, Any]) -> StandardModel:
    """Builds and trains GNN from scratch using configuration dictionary"""
    
    logger.info(f"features: {config['features']}")
    logger.info(f"truth: {config['truth']}")

    # Set up directory and naming conventions for saved results
    archive = os.path.join(config['base_dir'], "train_model_without_configs")
    run_name = f"dynedge_{config['target']}_{config['run_name_tag']}"

    # Create training and validation dataloaders
    train_dataloader, validate_dataloader = make_dataloaders(config)

    # Build model architecture
    model = build_model(config, train_dataloader)

    # Early stopping callback to prevent overfitting
    callbacks = [
        EarlyStopping(
            monitor="val_loss",  # Metric to monitor
            patience=config["early_stopping_patience"],  # Number of epochs with no improvement to stop
        ),
        # Optionally: ProgressBar()
    ]

    # Fit model to data
    model.fit(
        train_dataloader,
        validate_dataloader,
        callbacks=callbacks,
        **config["fit"],
    )
    return model

# Run model inference on test data and save results 
# Applying trained model to test data/batches (TESTING)
def inference(model, config: Dict[str, Any]) -> pd.DataFrame:
    """Applies model to the specified inference DB and saves results to disk."""

    # Create test dataloader (inference on full DB)
    test_dataloader = make_dataloader(
        db=config['inference_database_path'],
        selection=None,  # Full dataset
        pulsemaps=config['pulsemap'],
        features=features,
        truth=truth,
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        shuffle=False,
        labels={'direction': Direction()},
        index_column=config['index_column'],
        truth_table=config['truth_table'],
    )
    
    # Run inference and obtain DataFrame of predictions
    results = model.predict_as_dataframe(
        gpus=[0],
        dataloader=test_dataloader,
        prediction_columns=model.prediction_columns,
        additional_attributes=model.additional_attributes,
    )

    # Save predictions to structured folder path
    archive = os.path.join(config['base_dir'], "train_model_without_configs")
    run_name = f"dynedge_{config['target']}_{config['run_name_tag']}"
    db_name = config['path'].split("/")[-1].split(".")[0]
    path = os.path.join(archive, db_name, run_name)

    logger.info(f"Writing results to {path}")
    os.makedirs(path, exist_ok=True)
    results.to_csv(f"{path}/results.csv")

    return results



# Constants tell dataloader what features and labels to extract from SQLite database
features = FEATURES.KAGGLE
truth = TRUTH.KAGGLE

# Configuration
config = {
        "path": '/kaggle/working/batch_1.db',
        "inference_database_path": '/kaggle/working/batch_51.db',
        "pulsemap": 'pulse_table',
        "truth_table": 'meta_table',
        "features": features,
        "truth": truth,
        "index_column": 'event_id',
        "run_name_tag": 'my_example',
        "batch_size": 200,
        "num_workers": 4,
        "target": 'direction',
        "early_stopping_patience": 5,
        "fit": {
                "max_epochs": 50, # ADJUST
                "gpus": [0],
                "distribution_strategy": None,
                },
        'train_selection': '/kaggle/working/train_selection_max_200_pulses.csv',
        'validate_selection': '/kaggle/working/validate_selection_max_200_pulses.csv',
        'test_selection': None,
        'base_dir': 'training'
}


# Train from scratch (slow) - remember to save it!
model = train_dynedge_from_scratch(config = config)
model.save(path="mymodel3.pth")

# Load state-dict from pre-trained model (faster)
#model = load_pretrained_model(config = config)


# Inference
results = inference(model, config)


import pandas as pd
def convert_to_3d(df: pd.DataFrame) -> pd.DataFrame:
    """Converts zenith and azimuth to 3D direction vectors"""
    df['true_x'] = np.cos(df['azimuth']) * np.sin(df['zenith'])
    df['true_y'] = np.sin(df['azimuth'])*np.sin(df['zenith'])
    df['true_z'] = np.cos(df['zenith'])
    return df

def calculate_angular_error(df : pd.DataFrame) -> pd.DataFrame:
    """Calcualtes the opening angle (angular error) between true and reconstructed direction vectors"""
    df['angular_error'] = np.arccos(df['true_x']*df['direction_x'] + df['true_y']*df['direction_y'] + df['true_z']*df['direction_z'])
    return df

def average_angular_error(df : pd.DataFrame) -> pd.DataFrame:
    #Calculates average angular error across the results for an overall score
    return df['angular_error'].mean()


results = convert_to_3d(results)
results = calculate_angular_error(results)
avg_error = average_angular_error(results)
print("Average Angular Error Score:", avg_error)


fig = plt.figure(figsize = (6,6))
plt.hist(results['angular_error'], 
         bins = np.arange(0,np.pi*2, 0.05), 
         histtype = 'step', 
         label = f'mean angular error: {np.round(results["angular_error"].mean(),2)}')
plt.xlabel('Angular Error [rad.]', size = 15)
plt.ylabel('Counts', size = 15)
plt.title('Angular Error Distribution (Batch 51)', size = 15)
plt.legend(frameon = False, fontsize = 15)


# Uncertainty threshold for uncertainty sigma
cut_threshold = 0.5
fig = plt.figure(figsize = (6,6))
# Plot histogram for 'low-uncertainty' events 
plt.hist(results['angular_error'][1/np.sqrt(results['direction_kappa']) <= cut_threshold], 
         bins = np.arange(0,np.pi*2, 0.05), 
         histtype = 'step', 
         label = f'sigma <= {cut_threshold}: {np.round(results["angular_error"][1/np.sqrt(results["direction_kappa"]) <= cut_threshold].mean(),2)}')

# Plot histogram for 'high-uncertainty' events
plt.hist(results['angular_error'][1/np.sqrt(results['direction_kappa']) > cut_threshold], 
         bins = np.arange(0,np.pi*2, 0.05), 
         histtype = 'step', 
         label = f'sigma > {cut_threshold}: {np.round(results["angular_error"][1/np.sqrt(results["direction_kappa"]) > cut_threshold].mean(),2)}')
plt.xlabel('Angular Error [rad.]', size = 15)
plt.ylabel('Counts', size = 15)
plt.title('Angular Error Distribution (Batch 51)', size = 15)
plt.legend(frameon = False, fontsize = 15)


# Reading contents of batch files .db
import sqlite3

# Connect to the database
conn = sqlite3.connect("/kaggle/input/batch-51/batch_51.db")
cursor = conn.cursor()

# List all tables
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()
print("Tables:", tables)

# Pick the first table (or loop through them)
first_table = tables[0][0]

# Get column names (table schema)
cursor.execute(f"PRAGMA table_info({first_table});")
columns = cursor.fetchall()
print("Columns:")
for col in columns:
    print(col)

# Show first few rows
cursor.execute(f"SELECT * FROM {first_table} LIMIT 20;")
rows = cursor.fetchall()
print("First few rows:")
for row in rows:
    print(row)

# Pick the first table (or loop through them)
first_table = tables[1][0]

# Get column names (table schema)
cursor.execute(f"PRAGMA table_info({first_table});")
columns = cursor.fetchall()
print("Columns:")
for col in columns:
    print(col)

# Show first few rows
cursor.execute(f"SELECT * FROM {first_table} LIMIT 100;")
rows = cursor.fetchall()
print("First few rows:")
for row in rows:
    print(row)

conn.close()


import sqlite3
import pandas as pd
import matplotlib.pyplot as plt

# Connect to the database
conn = sqlite3.connect("/kaggle/input/batch-51/batch_51.db")

# Load second table (index 1)
tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table';", conn)
second_table = tables.iloc[1, 0]

# Load relevant columns into a DataFrame
df = pd.read_sql(f"SELECT event_id, charge FROM {second_table}", conn)
conn.close()

# Group by event_id and compute average charge
avg_charge = df.groupby("event_id")["charge"].mean().reset_index()

# Compute and print the median average charge
median_avg_charge = avg_charge["charge"].median()
print(f"Median average charge across events: {median_avg_charge:.3f}")

# Sort by event_id
avg_charge_sorted = avg_charge.sort_values("event_id")

# Plotting
plt.figure(figsize=(10, 5))
plt.plot(avg_charge_sorted["event_id"], avg_charge_sorted["charge"], marker='o', linestyle='-', markersize=2)
plt.axhline(median_avg_charge, color='red', linestyle='--', label=f"Median = {median_avg_charge:.2f}")
plt.xlabel("Event ID")
plt.ylabel("Average Charge")
plt.title(f"Average Charge per Event in {second_table}")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()



import sqlite3

# Open original database
conn = sqlite3.connect("/kaggle/input/batch-51/batch_51.db")
cursor = conn.cursor()

# Create new databases
conn_high = sqlite3.connect("/kaggle/working/batch_51_high.db")
cursor_high = conn_high.cursor()

conn_low = sqlite3.connect("/kaggle/working/batch_51_low.db")
cursor_low = conn_low.cursor()

# Create tables in new databases
cursor.execute("SELECT sql FROM sqlite_master WHERE type='table';")
table_creations = cursor.fetchall()

for create_sql in table_creations:
    cursor_high.execute(create_sql[0])
    cursor_low.execute(create_sql[0])

# Find all event_ids
cursor.execute("SELECT event_id FROM meta_table;")
all_event_ids = [row[0] for row in cursor.fetchall()]

for event_id in all_event_ids:
    # Get all pulses for this event
    cursor.execute("SELECT charge FROM pulse_table WHERE event_id = ?", (event_id,))
    charges = [row[0] for row in cursor.fetchall()]

    if not charges:
        continue  # Skip if no pulses

    avg_charge = sum(charges) / len(charges)

    # Fetch corresponding meta_table entry
    cursor.execute("SELECT * FROM meta_table WHERE event_id = ?", (event_id,))
    meta_entry = cursor.fetchone()

    # Fetch corresponding pulse_table entries
    cursor.execute("SELECT * FROM pulse_table WHERE event_id = ?", (event_id,))
    pulse_entries = cursor.fetchall()

    # Insert into appropriate database
    if avg_charge > 0.95:
        cursor_high.execute("INSERT INTO meta_table VALUES (?, ?, ?, ?, ?, ?)", meta_entry)
        cursor_high.executemany("INSERT INTO pulse_table VALUES (?, ?, ?, ?, ?, ?, ?, ?)", pulse_entries)
    else:
        cursor_low.execute("INSERT INTO meta_table VALUES (?, ?, ?, ?, ?, ?)", meta_entry)
        cursor_low.executemany("INSERT INTO pulse_table VALUES (?, ?, ?, ?, ?, ?, ?, ?)", pulse_entries)

# Commit and close all
conn_high.commit()
conn_low.commit()
conn_high.close()
conn_low.close()
conn.close()


import sqlite3
import pandas as pd

# Connect to the database
conn = sqlite3.connect('/kaggle/working/batch_51_high.db')

# List all tables
tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table';", conn)
print("Tables in database:")
print(tables)

# Choose the first table (assuming there's one main table)
first_table = tables.iloc[1, 0]

# Show the first few rows
df_preview = pd.read_sql_query(f"SELECT * FROM {first_table} LIMIT 5;", conn)
print("\nFirst 5 rows from the table:")
print(df_preview)

# Don't forget to close the connection
conn.close()



import sqlite3

# Open original database
conn = sqlite3.connect("/kaggle/input/batch-51/batch_51.db")
cursor = conn.cursor()

# Create new databases
conn_high = sqlite3.connect("/kaggle/working/batch_51_high_auxzero.db")
cursor_high = conn_high.cursor()

conn_low = sqlite3.connect("/kaggle/working/batch_51_low_auxzero.db")
cursor_low = conn_low.cursor()

# Get table creation scripts from the original database
cursor.execute("SELECT sql FROM sqlite_master WHERE type='table';")
table_creations = cursor.fetchall()

# Create tables in the new databases
for create_sql in table_creations:
    try:
        cursor_high.execute(create_sql[0])
        cursor_low.execute(create_sql[0])
    except sqlite3.OperationalError:
        # Skip if the table already exists
        pass

# Find all event_ids from the meta_table
cursor.execute("SELECT event_id FROM meta_table;")
all_event_ids = [row[0] for row in cursor.fetchall()]

for event_id in all_event_ids:
    # Get all auxiliary values for this event from the pulse_table
    cursor.execute("SELECT auxiliary FROM pulse_table WHERE event_id = ?", (event_id,))
    auxiliary_flags = [row[0] for row in cursor.fetchall()]

    if not auxiliary_flags:
        continue  # Skip if no auxiliary flags for the event

    # Calculate percentage of zeros in auxiliary flags
    zero_percentage = auxiliary_flags.count(0) / len(auxiliary_flags) * 100

    # Fetch corresponding meta_table entry
    cursor.execute("SELECT * FROM meta_table WHERE event_id = ?", (event_id,))
    meta_entry = cursor.fetchone()

    # Fetch corresponding pulse_table entries
    cursor.execute("SELECT * FROM pulse_table WHERE event_id = ?", (event_id,))
    pulse_entries = cursor.fetchall()

    # Insert into the appropriate database based on zero percentage
    if zero_percentage > 70:
        cursor_high.execute("INSERT INTO meta_table VALUES (?, ?, ?, ?, ?, ?)", meta_entry)
        cursor_high.executemany("INSERT INTO pulse_table VALUES (?, ?, ?, ?, ?, ?, ?, ?)", pulse_entries)
    else:
        cursor_low.execute("INSERT INTO meta_table VALUES (?, ?, ?, ?, ?, ?)", meta_entry)
        cursor_low.executemany("INSERT INTO pulse_table VALUES (?, ?, ?, ?, ?, ?, ?, ?)", pulse_entries)

# Commit and close all connections
conn_high.commit()
conn_low.commit()
conn_high.close()
conn_low.close()
conn.close()

print("Batch 51 DB split by auxiliary flag percentage complete!")



import sqlite3

# Open original database
conn = sqlite3.connect("/kaggle/input/batch-51/batch_51.db")
cursor = conn.cursor()

# Query to get all table names
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()

# Print out table names to check what exists in the database
for table in tables:
    print(table[0])

# Close connection
conn.close()





