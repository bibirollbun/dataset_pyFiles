from IPython.core.display import HTML


def apply_css_styles(
    font_name: str = "Lato",
    fallback_font: str = "Verdana",
    css_content: str = None,
    verbose: bool = False
) -> HTML:
    """Applies custom CSS styles within a Jupyter notebook cell.

    Args:
        font_name (str, optional): 
            The primary font to use in the styles.
        fallback_font (str, optional): 
            The fallback font to use if the primary font is unavailable.
        css_content (str, optional): 
            Custom CSS content to use. 
            If None, default styles are used.
        verbose (bool, optional): 
            Whether to print the generated CSS for debugging.

    Returns:
        IPython.core.display.HTML: 
            HTML object with the injected styles.
    """
    try:
        # Default CSS content if none is provided
        default_css = '''
p, li, a, b, h1, h2, h3, h4, h5, h6, title, ul, strong, sup, sub, em, i, blockquote, label {
    font-family: Verdana !important;
}

b, h1 {
    font-weight: 900 !important;
}

h2, h3, h4 ul {
    font-weight: 700 !important;
}

.fa, .far, .fas {
    font-family: "Font Awesome 5 Free" !important;
}
'''

        # Generate font import string dynamically based on the provided font name
        font_import = (
            f"\n@import url('https://fonts.googleapis.com/css2?family={font_name.replace(' ', '+')}:ital,wght@0,100;0,300;0,400;0,700;0,900;1,100;1,300;1,400;1,700;1,900&display=swap');\n"
        )

        # Use provided CSS content or fallback to default
        css_to_use = css_content or default_css

        # Replace fallback font in the CSS content
        css_to_use = css_to_use.replace("Verdana", font_name)

        # Combine the font import and the CSS content into a single HTML style block
        combined_styles = f"<style>{font_import}{css_to_use}</style>"

        if verbose:
            print(combined_styles)  # Print the CSS for debugging if verbose is True

        return HTML(combined_styles)  # Return the generated styles as an HTML object

    except Exception as e:
        raise RuntimeError(f"An error occurred while applying styles: {str(e)}")

# Apply styles (example usage)
apply_css_styles(verbose=False)


print("\n... PIP INSTALLS STARTING ...\n")
print("\n... PIP INSTALLS COMPLETE ...\n")

print("\n... IMPORTS STARTING ...\n")
print("\n\tVERSION INFORMATION")
import pandas as pd; pd.options.mode.chained_assignment = None; pd.set_option('display.max_columns', None);
import numpy as np; print(f"\t\tâ€“ NUMPY VERSION: {np.__version__}");
import sklearn; print(f"\t\tâ€“ SKLEARN VERSION: {sklearn.__version__}");

# Built-In Imports (mostly don't worry about these)
from typing import Iterable, Any, Callable, Generator
from kaggle_datasets import KaggleDatasets
from dataclasses import dataclass
from collections import Counter
from datetime import datetime
from zipfile import ZipFile
from glob import glob
import subprocess
import warnings
import requests
import textwrap
import hashlib
import imageio
import IPython
import urllib
import zipfile
import pickle
import random
import shutil
import string
import yaml
import json
import copy
import math
import time
import gzip
import ast
import sys
import io
import gc
import re
import os

# Visualization Imports (overkill)
from IPython.core.display import HTML, Markdown
from matplotlib.patches import Rectangle
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm; tqdm.pandas();
from mpl_toolkits.mplot3d import Axes3D
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from PIL import Image, ImageEnhance; Image.MAX_IMAGE_PIXELS = 5_000_000_000;
import matplotlib; print(f"\t\tâ€“ MATPLOTLIB VERSION: {matplotlib.__version__}");
import plotly
import PIL

# Rich
import rich
from rich import pretty; pretty.install()
from rich.markdown import Markdown
from rich import print as rprint
from rich.console import Console
from rich.style import Style
from rich.live import Live
from rich.text import Text
from rich import inspect

def seed_it_all(seed=7):
    """ Attempt to be Reproducible """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    # tf.random.set_seed(seed)
    
seed_it_all()

warnings.filterwarnings('ignore', category=FutureWarning, message='use_inf_as_na option is deprecated')
print("\n\n... IMPORTS COMPLETE ...\n")


"""The code below is heavily inspired by the notebook: https://www.kaggle.com/code/andrewjdarley/parse-data

    (1) I first understood Andrew's code, then I rewrote it to be more aligned with my own style.
    (2) Next, I updated it to incorporate any changes I think are appropriate.
    (3) Last, in the cell in section 6, I wrap this into a single unified function to run the prep end to end.
"""


def create_dataset_directories(
    yolo_images_train: str, 
    yolo_images_val: str, 
    yolo_labels_train: str, 
    yolo_labels_val: str
) -> None:
    """
    Create all necessary directories for the YOLO dataset.
    
    Args:
        yolo_images_train (str): Path to directory for training images
        yolo_images_val (str): Path to directory for validation images
        yolo_labels_train (str): Path to directory for training labels
        yolo_labels_val (str): Path to directory for validation labels
    """
    for dir_path in [yolo_images_train, yolo_images_val, yolo_labels_train, yolo_labels_val]:
        os.makedirs(dir_path, exist_ok=True)
        print(f"CREATED DIRECTORY\n\t--> {dir_path}")


def normalize_slice(slice_data: np.ndarray) -> np.ndarray:
    """Normalize a tomographic slice for better visualization and learning.
    
    Uses percentile-based normalization to enhance contrast while 
    preserving important features.
    
    Args:
        slice_data (np.ndarray): 
            Raw numpy array of the tomographic slice
        
    Returns:
        np.ndarray:
            Normalized slice data as uint8 numpy array (0-255 range)
    """
    # (1) Calculate 2nd and 98th percentiles for robust normalization
    p2 = np.percentile(slice_data, 2)
    p98 = np.percentile(slice_data, 98)
    
    # (2) Clip the data to the percentile range to reduce outlier influence
    clipped_data = np.clip(slice_data, p2, p98)
    
    # (3) Normalize to [0, 255] range for standard image representation
    normalized = 255 * (clipped_data - p2) / (p98 - p2)
    
    # (4) Convert to 8-bit unsigned integer format for image saving
    return np.uint8(normalized)
    

def validate_labels_file(labels_path: str) -> pd.DataFrame:
    """Load and validate the labels CSV file.
    
    Args:
        labels_path (str): Path to the labels CSV file
        
    Returns:
        pd.DataFrame:
            The processed label information.
        
    Raises:
        FileNotFoundError: If labels file doesn't exist
        ValueError: If labels file format is invalid
    """
    # (0) Define the required columns
    _required_columns = [
        'tomo_id', 
        'Motor axis 0', 
        'Motor axis 1', 
        'Motor axis 2',
        'Array shape (axis 0)', 
        'Number of motors'
    ]
    
    # (1) Check if labels file exists
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Labels file not found: {labels_path}")
    
    # (2) Load the labels CSV
    try:
        labels_df = pd.read_csv(labels_path)
    except Exception as e:
        raise ValueError(f"Error reading labels file: {e}")
    
    # (3) Validate required columns exist
    missing_columns = [col for col in _required_columns if col not in labels_df.columns]
    if missing_columns:
        raise ValueError(f"Labels file missing required columns: {', '.join(missing_columns)}")
        
    # (4) Return the validated DataFrame
    return labels_df


def split_tomograms(
    labels_df: pd.DataFrame, 
    train_split: float = 0.8,
    random_seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Split tomograms into training and validation sets.
    
    Performs split at the tomogram level to ensure all slices from a single
    tomogram are in the same set (train or validation).
    
    Args:
        labels_df (pd.DataFrame): 
            DataFrame containing the tomogram labels
        train_split (float, optional): 
            Fraction of data to use for training (0.0-1.0)
        random_seed (int, optional): 
            Random seed for reproducibility
        
    Returns:
        tuple[np.ndarray, np.ndarray]:
            The tomograms split into their respective train and validation distributions.
    """
    # (1) Set random seed for reproducibility
    np.random.seed(random_seed)
    
    # (2) Find tomograms that have motors (tomograms of interest)
    tomo_df = labels_df[labels_df['Number of motors'] > 0].copy()
    unique_tomos = tomo_df['tomo_id'].unique()
    
    print(f"\nFOUND {len(unique_tomos)} UNIQUE TOMOGRAMS WITH 1 OR MORE MOTORS\n")
    
    # (3) Shuffle tomograms for random split
    np.random.shuffle(unique_tomos)
    
    # (4) Calculate split index based on train_split ratio
    split_idx = int(len(unique_tomos) * train_split)
    
    # (5) Create train and validation sets
    train_tomos = unique_tomos[:split_idx]
    val_tomos = unique_tomos[split_idx:]
    
    print(f"\nSPLIT DISTRIBUTION:\n\tTRAIN: {len(train_tomos)} TOMOGRAMS\n\tVALIDATION: {len(val_tomos)} TOMOGRAMS.")
    
    return train_tomos, val_tomos

def process_tomogram_set(
    labels_df: pd.DataFrame,
    tomogram_ids: np.ndarray, 
    train_dir: str,
    images_dir: str, 
    labels_dir: str, 
    set_name: str,
    trust: int = 4,
    box_size: int = 24
) -> tuple[int, int]:
    """Process a set of tomograms, extracting slices and creating annotations.
    
    Args:
        labels_df (pd.DataFrame): DataFrame containing the tomogram labels
        tomogram_ids (np.ndarray): Array of tomogram IDs to process
        train_dir (str): Directory containing the raw tomogram data
        images_dir (str): Directory to save processed images
        labels_dir (str): Directory to save annotation labels
        set_name (str): Name of the dataset (e.g., "training" or "validation")
        trust (int, optional): Number of slices above and below center slice to include
        box_size (int, optional): Size of bounding box in pixels for annotations
        
    Returns:
        tuple[int, int]:
            The count of slices and the count of motors.
    """
    # (1) Extract motor information for the specified tomograms
    motor_info = []
    for tomo_id in tomogram_ids:
        # Get all motors for this tomogram
        tomo_motors = labels_df[labels_df['tomo_id'] == tomo_id]
        for _, motor in tomo_motors.iterrows():
            if pd.isna(motor['Motor axis 0']):
                continue
            motor_info.append(
                (tomo_id, 
                 int(motor['Motor axis 0']), 
                 int(motor['Motor axis 1']), 
                 int(motor['Motor axis 2']),
                 int(motor['Array shape (axis 0)']))
            )
    
    # (2) Output processing information
    print(f"\nPROCESSING APPROXIMATELY {len(motor_info) * (2 * trust + 1)} SLICES FOR '{set_name}'\n")
    
    # (3) Process each motor
    processed_slices = 0
    
    # (4) Process all motors across all tomograms in the set
    for tomo_id, z_center, y_center, x_center, z_max in tqdm(motor_info, desc=f"PROCESSING {set_name} MOTORS"):
        # (4.1) Calculate range of slices to include based on trust parameter
        z_min = max(0, z_center - trust)
        z_max = min(z_max - 1, z_center + trust)
        
        # (4.2) Process each slice in the defined range
        for z in range(z_min, z_max + 1):
            # Create slice filename
            slice_filename = f"slice_{z:04d}.jpg"
            
            # Source path for the slice
            src_path = os.path.join(train_dir, tomo_id, slice_filename)
            
            # (4.3) Skip if source file doesn't exist
            if not os.path.exists(src_path):
                print(f"Warning: {src_path} does not exist, skipping.")
                continue
            
            # (4.4) Load and normalize the slice
            try:
                img = Image.open(src_path)
                img_array = np.array(img)
            except Exception as e:
                print(f"Error loading image {src_path}: {e}")
                continue
            
            # (4.5) Normalize the image
            try:
                normalized_img = normalize_slice(img_array)
            except Exception as e:
                print(f"Error normalizing image {src_path}: {e}")
                continue
            
            # (4.6) Create destination filename with unique identifier
            dest_filename = f"{tomo_id}_z{z:04d}_y{y_center:04d}_x{x_center:04d}.jpg"
            dest_path = os.path.join(images_dir, dest_filename)
            
            # (4.7) Save the normalized image
            try:
                Image.fromarray(normalized_img).save(dest_path)
            except Exception as e:
                print(f"Error saving image {dest_path}: {e}")
                continue
            
            # (4.8) Get image dimensions for normalization
            img_width, img_height = img.size
            
            # (4.9) Create YOLO format label (see below)
            #    - <class> <x_center> <y_center> <width> <height>
            #    - Values are normalized to [0, 1]
            x_center_norm = x_center / img_width
            y_center_norm = y_center / img_height
            box_width_norm = box_size / img_width
            box_height_norm = box_size / img_height
            
            # (4.10) Write label file
            label_path = os.path.join(labels_dir, dest_filename.replace('.jpg', '.txt'))
            try:
                with open(label_path, 'w') as f:
                    f.write(f"0 {x_center_norm} {y_center_norm} {box_width_norm} {box_height_norm}\n")
            except Exception as e:
                print(f"Error writing label {label_path}: {e}")
                continue
            
            # (4.11) Increment slice counter
            processed_slices += 1
    
    # (5) Return statistics
    return processed_slices, len(motor_info)


def create_yaml_config(yolo_dataset_dir: str) -> str:
    """Create YAML configuration file for YOLO training.
    
    Args:
        yolo_dataset_dir (str): Base directory for the YOLO dataset
        
    Returns:
        str: The path to the created YAML file
    """
    # (1) Define YAML content with dataset paths and class names
    yaml_content = {
        'path': yolo_dataset_dir,
        'train': 'images/train',
        'val': 'images/val',
        'names': {0: 'motor'}
    }
    
    # (2) Define output path
    yaml_path = os.path.join(yolo_dataset_dir, 'dataset.yaml')
    
    # (3) Write YAML file
    try:
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_content, f, default_flow_style=False)
    except Exception as e:
        print(f"Warning: Failed to write YAML config: {e}")
        
    # (4) Return path to the created file
    return yaml_path


def flatten_l_o_l(nested_iterable: Iterable[Iterable[Any]]) -> list[Any]:
    """Flatten a list of lists (or any nested iterable) into a single list.
    
    Transforms a nested structure like [[1, 2], [3, 4]] into [1, 2, 3, 4].
    
    Args:
        nested_iterable (Iterable[Iterable[Any]]): 
            An iterable containing other iterables to be flattened.
            Examples: List of lists, tuple of sets, etc.
    
    Returns:
        list[T]: A flattened list containing all items from the input nested structure.
    
    Examples:
        >>> flatten_l_o_l([[1, 2], [3, 4]])
        [1, 2, 3, 4]
        >>> flatten_l_o_l([(5, 6), [7, 8]])
        [5, 6, 7, 8]
    """
    # (1) Use list comprehension with nested loops to flatten the structure
    return [item for sublist in nested_iterable for item in sublist]


def print_ln(
    symbol: str = "-", 
    line_len: int = 110, 
    newline_before: bool = False, 
    newline_after: bool = False
) -> None:
    """Print a horizontal line of a specified length and symbol.
    
    Creates a visual separator in console output for improved readability.
    
    Args:
        symbol: The character(s) to use for the horizontal line.
            Single character strings work best (e.g., "-", "=", "*").
        line_len: The length of the horizontal line in characters.
            Default is 110 characters.
        newline_before: Whether to print a newline character before the line.
            Used to create spacing before the separator.
        newline_after: Whether to print a newline character after the line.
            Used to create spacing after the separator.
            
    Returns:
        None: This function prints to stdout but doesn't return any value.
    
    Examples:
        >>> print_ln()  # Prints "----------..." (110 dashes)
        >>> print_ln("=", 50, True, True)  # Prints a newline, then 50 "=" characters, then a newline
    """
    # (1) Print a newline before the line if requested
    if newline_before:
        print()
    
    # (2) Print the line using string multiplication
    print(symbol * line_len)
    
    # (3) Print a newline after the line if requested
    if newline_after:
        print()
        
        
def display_hr(
    newline_before: bool = False, 
    newline_after: bool = False
) -> None:
    """Display an HTML horizontal rule (<hr>) in notebook environments.
    
    Creates a visual separator in Jupyter/IPython notebook output.
    
    Args:
        newline_before: Whether to print a newline character before the horizontal rule.
            Used to create spacing before the separator.
        newline_after: Whether to print a newline character after the horizontal rule.
            Used to create spacing after the separator.
            
    Returns:
        None: This function displays HTML content but doesn't return any value.
    
    Notes:
        - This function is designed for use in Jupyter notebook or IPython environments.
        - It will not render correctly in standard console environments.
    
    Examples:
        >>> display_hr()  # Displays an HTML horizontal rule
        >>> display_hr(True, True)  # Displays a newline, then an HTML horizontal rule, then a newline
    """
    # (1) Print a newline before the HTML horizontal rule if requested
    if newline_before:
        print()
    
    # (2) Display the HTML horizontal rule
    display(HTML("<hr>"))
    
    # (3) Print a newline after the HTML horizontal rule if requested
    if newline_after:
        print()


def wrap_text(text: str, width: int = 88) -> str:
    """Wrap text to a specified width.
    
    Formats a long string by inserting line breaks to ensure no line exceeds
    the specified width. Useful for formatting paragraphs for display in
    fixed-width contexts.
    
    Args:
        text: The text string to wrap.
            Can be a single line or multiple lines.
        width: The maximum width of a line in characters.
            Default is 88 characters, which matches Black formatter's default.

    Returns:
        str: The wrapped text with added line breaks.
    
    Examples:
        >>> long_text = "This is a very long string that needs to be wrapped to multiple lines."
        >>> wrap_text(long_text, 20)
        'This is a very long\\nstring that needs to\\nbe wrapped to\\nmultiple lines.'
    """
    # (1) Use textwrap.fill to wrap the text to the specified width
    return textwrap.fill(text, width)


def wrap_text_by_paragraphs(text: str, width: int = 88) -> str:
    """Wrap text by paragraphs to a specified width while preserving paragraph structure.
    
    Similar to wrap_text(), but maintains paragraph separation by preserving
    blank lines between paragraphs.
    
    Args:
        text: The text string containing multiple paragraphs to wrap.
            Paragraphs should be separated by newline characters.
        width: The maximum width of a line in characters.
            Default is 88 characters, which matches Black formatter's default.

    Returns:
        str: The wrapped text with preserved paragraph separation.
    
    Examples:
        >>> paragraphs = "First paragraph.\\n\\nSecond paragraph that is longer."
        >>> wrap_text_by_paragraphs(paragraphs, 20)
        'First paragraph.\\n\\nSecond paragraph\\nthat is longer.'
    """
    # (1) Split the text into paragraphs using newline characters
    paragraphs = text.split('\n')
    
    # (2) Wrap each paragraph individually
    wrapped_paragraphs = [textwrap.fill(paragraph, width) for paragraph in paragraphs]
    
    # (3) Join the wrapped paragraphs with double newlines to preserve paragraph structure
    return '\n\n'.join(wrapped_paragraphs)


# ROOT PATHS (define these in your notebook)
WORKING_DIR = "/kaggle/working"
INPUT_DIR = "/kaggle/input"
COMPETITION_DIR = os.path.join(INPUT_DIR, "byu-locating-bacterial-flagellar-motors-2025")

# COMPETITION DATA PATHS
TRAIN_DIR = os.path.join(COMPETITION_DIR, "train")
TRAIN_LABELS_PATH = os.path.join(COMPETITION_DIR, "train_labels.csv")
TEST_DIR = os.path.join(COMPETITION_DIR, "test")

# OUTPUT PATHS
YOLO_DATASET_DIR = os.path.join(WORKING_DIR, "yolo_dataset")
YOLO_IMAGES_TRAIN = os.path.join(YOLO_DATASET_DIR, "images", "train")
YOLO_IMAGES_VAL = os.path.join(YOLO_DATASET_DIR, "images", "val")
YOLO_LABELS_TRAIN = os.path.join(YOLO_DATASET_DIR, "labels", "train")
YOLO_LABELS_VAL = os.path.join(YOLO_DATASET_DIR, "labels", "val")

# DATASET PROCESSING HYPERPARAMETERS
TRUST = 4          # Number of slices above and below center slice
BOX_SIZE = 24      # Bounding box size for annotations
TRAIN_SPLIT = 0.8  # 80% for training, 20% for validation

# LOAD THE DATASET
labels_df = validate_labels_file(TRAIN_LABELS_PATH)

rich.print("\n\n[bold red]LABELS DATAFRAME[/bold red]")
labels_df


labels_df.info()
labels_df.describe().T


def visualize_motor_counts(
    df: pd.DataFrame,
    color_sequence: list[str] | None = None,
    height: int = 500,
    width: int = 800
) -> px.pie:
    """
    Creates a pie chart visualization showing the distribution of tomograms 
    by their motor count.
    
    Args:
        df (pd.Dataframe): The tomogram dataset.
        color_sequence (list[str], optional): Colors for the pie chart (default: Plotly default)
        height (int, optional): Height of the figure in pixels.
        width (int, optional): Width of the figure in pixels.
        
    Returns:
        A Plotly pie chart figure showing distribution of tomograms by motor count
        
    Example:
        >>> fig = visualize_motor_counts(labels_df)
        >>> fig.show(renderer="iframe")
    """
    # (1) Group by tomo_id and get the number of motors for each unique tomogram
    # We only need one row per tomogram since 'Number of motors' is the same for all rows of the same tomogram
    motors_per_tomo = df.drop_duplicates(subset=['tomo_id'])[['tomo_id', 'Number of motors']]
    
    # (2) Count tomograms by their motor count
    motor_count_distribution = motors_per_tomo['Number of motors'].value_counts().reset_index()
    motor_count_distribution.columns = ['motor_count', 'num_tomograms']
    
    # (3) Sort by motor count for better interpretation
    motor_count_distribution = motor_count_distribution.sort_values('motor_count')
    
    # (4) Create the pie chart
    fig = px.pie(
        motor_count_distribution, values='num_tomograms', names='motor_count',  # Information to plot
        title='<b>Distribution of Tomograms by Motor Count</b>',                # Title
        color_discrete_sequence=color_sequence,                                 # Colour Sequence
        height=height, width=width, hole=0.3,                                   # Sizing and Whatnot
        category_orders={"motor_count": sorted(motor_count_distribution['motor_count'].tolist())},
    )
    
    # (5) Improve layout for better readability
    fig.update_layout(
        margin=dict(l=20, r=120, t=100, b=20),  # Increased right margin for legend
        legend_title='<b>Number of Motors</b>',  # Bold legend title
    )
    
    # (6) Add percentage and count to hover information and adjust text position
    fig.update_traces(
        textinfo='percent+label',
        texttemplate='<b>%{label}</b><br>%{percent}',  # Bold labels
        textposition='inside',                         # This avoids the callout interfering with the title.
        hovertemplate='<b>Number of Motors: %{label}</b><br>Number of Tomograms: %{value}<br>Percentage: %{percent}'
    )
    
    return fig


motor_counts_fig = visualize_motor_counts(labels_df)
motor_counts_fig.show(renderer='iframe')


def visualize_tomo_distribution(
    df: pd.DataFrame, 
    n_top: int = 10, 
    color: str = '#d27582',
    height: int = 700,
    width: int = 900
) -> px.bar:
    """Visualizes the distribution of tomogram IDs in the dataset.
    
    This function counts the frequency of each tomogram ID and plots the top N most 
    frequently occurring tomograms, which correlates with those containing multiple motors.
    
    Args:
        df (pd.DataFrame): DataFrame containing the tomogram dataset with at least 'tomo_id' column
        n_top (int, optional): Number of top tomograms to display (default: 15)
        color (str, optional): Color for the bar chart (default: #d27582)
        height (int, optional): Height of the figure in pixels (default: 500)
        width (int, optional): Width of the figure in pixels (default: 800)
        
    Returns:
        A Plotly bar chart figure showing tomogram ID frequencies
        
    Example:
        >>> fig = visualize_tomo_distribution(labels_df)
        >>> fig.show(renderer='iframe')
    """
    # (1) Count the frequency of each tomogram ID
    tomo_counts = df['tomo_id'].value_counts().reset_index()
    tomo_counts.columns = ['tomo_id', 'count']
    
    # (2) Sort by count in descending order
    tomo_counts = tomo_counts.sort_values('count', ascending=False)
    
    # (3) Subset to only include the top n_top tomograms
    top_tomos = tomo_counts.head(n_top)
    
    # (4) Create a horizontal bar chart for better readability of tomogram IDs
    fig = px.bar(
        top_tomos, 
        y='tomo_id', 
        x='count',
        orientation='h',
        color_discrete_sequence=[color],
        title=f'<b>Top {n_top} Tomograms by Number of Motors</b>',  # Bold title
        labels={'count': 'Number of Motors', 'tomo_id': 'Tomogram ID'},
        height=height,
        width=width
    )
    
    # (5) Improve layout for better readability
    fig.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        xaxis_title='<b>Number of Motors</b>',  # Bold axis title
        yaxis_title='<b>Tomogram ID</b>',  # Bold axis title
        margin=dict(l=40, r=40, t=80, b=40),  # Increased margins for better spacing
        title=dict(
            text=f'<b>Top {n_top} Tomograms by Number of Motors</b>',
            font=dict(size=22)  # Larger title
        ),
        title_x=0.5,  # Center the title
        title_y=0.95,  # Position title higher
        font=dict(family="Arial, sans-serif"),  # Consistent font family
    )
    
    # (6) Add data labels on bars and customize hover information
    fig.update_traces(
        texttemplate='<b>%{x}</b>',  # Bold text showing count
        textposition='outside',  # Position text outside bars
        textfont=dict(size=12, color="black"),  # Text formatting
        hovertemplate='<b>Tomogram ID:</b> %{y}<br><b>Number of Motors:</b> %{x}<extra></extra>'
    )
    
    # (7) Apply consistent styling to axes
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgray',
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor='black',
        tickfont=dict(size=12),
    )
    
    fig.update_yaxes(
        tickfont=dict(size=12),
        tickmode='linear'  # Ensure all ticks are shown
    )
    
    return fig


tomo_distribution_fig = visualize_tomo_distribution(labels_df, 25)
tomo_distribution_fig.show(renderer='iframe')


def visualize_motor_axis_distribution(
    df: pd.DataFrame,
    axis: int,
    color: str = '#d27582',
    bin_width: int | None = None,
    height: int = 500,
    width: int = 800
) -> go.Figure:
    """Creates a histogram visualization showing the distribution of motor positions.
    
    Must be done along a specified axis (0: z, 1: y, 2: x).
    
    Args:
        df (pd.DataFrame): 
            The tomogram dataset with at least 'Motor axis 0', 'Motor axis 1', 'Motor axis 2' columns
        axis (int): 
            The axis to visualize (0 for z, 1 for y, 2 for x)
        color (str, optional):
            Color for the histogram bars (default: '#d27582')
        bin_width (int, optional):
            Width of histogram bins. If None, automatically determined (default: None)
        height (int, optional): 
            Height of the figure in pixels (default: 500)
        width (int, optional): 
            Width of the figure in pixels (default: 800)
        
    Returns:
        A Plotly histogram figure showing distribution of motor positions along the specified axis
        
    Example:
        >>> fig = visualize_motor_axis_distribution(labels_df, axis=0)
        >>> fig.show(renderer='iframe')
    """
    # Filter out rows with -1 values (no motor present)
    filtered_df = df[df[f'Motor axis {axis}'] >= 0]
    
    # Define axis labels
    axis_names = {0: 'Z (Slice)', 1: 'Y', 2: 'X'}
    axis_label = axis_names[axis]
    
    # Determine bin width if not specified
    if bin_width is None:
        range_of_values = filtered_df[f'Motor axis {axis}'].max() - filtered_df[f'Motor axis {axis}'].min()
        bin_width = max(1, round(range_of_values / 30))  # Default to 30 bins, minimum width of 1
    
    # Create histogram
    fig = px.histogram(
        filtered_df, 
        x=f'Motor axis {axis}',
        nbins=None,  # Let Plotly determine bins based on bin_width
        histnorm=None,  # Count values directly
        color_discrete_sequence=[color],
        title=f'<b>Distribution of Motor Positions - {axis_label} Axis</b>',
        labels={f'Motor axis {axis}': f'{axis_label} Position'},
        height=height,
        width=width
    )
    
    # Update layout for better readability
    fig.update_layout(
        xaxis_title=f'<b>{axis_label} Position</b>',
        yaxis_title='<b>Count</b>',
        margin=dict(l=40, r=40, t=80, b=40),
        title=dict(
            font=dict(size=22),  # Larger title
        ),
        title_x=0.5,  # Center the title
        title_y=0.95,  # Position title higher
        font=dict(family="Arial, sans-serif"),
        bargap=0.1  # Gap between bars
    )
    
    # Customize axis appearance
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgray',
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor='black',
        tickfont=dict(size=12)
    )
    
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgray',
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor='black',
        tickfont=dict(size=12)
    )
    
    # Enhance hover information
    fig.update_traces(
        hovertemplate=f'<b>{axis_label} Position</b>: %{{x}}<br><b>Count</b>: %{{y}}<extra></extra>'
    )
    
    return fig


# Visualize the Z-axis (slice) distribution
z_dist = visualize_motor_axis_distribution(labels_df, axis=0)
z_dist.show(renderer='iframe')

# Visualize the Y-axis distribution
y_dist = visualize_motor_axis_distribution(labels_df, axis=1)
y_dist.show(renderer='iframe')

# Visualize the X-axis distribution
x_dist = visualize_motor_axis_distribution(labels_df, axis=2)
x_dist.show(renderer='iframe')


def visualize_motor_3d_distribution(
    df: pd.DataFrame,
    color_by: str = 'tomogram_group',
    opacity: float = 0.7,
    marker_size: int = 5,
    height: int = 700,
    width: int = 900
) -> go.Figure:
    """Creates a 3D scatter plot visualization.
    
    This will show the distribution of motor positions in 3D space (x, y, z coordinates).
    
    Args:
        df (pd.DataFrame): 
            The tomogram dataset with at least 'Motor axis 0', 'Motor axis 1', 'Motor axis 2' columns
        color_by (str, optional):
            Attribute to use for coloring points. Options:
            - 'tomogram_group': Group tomograms by motor count (1, 2-3, 4+)
            - 'z_position': Color based on Z-axis position (slice number)
            - 'voxel_spacing': Color based on tomogram resolution
            - 'tomo_id': Color based on tomogram ID (not recommended for many tomograms)
            - 'motor_count': Original coloring by number of motors in tomogram
            (default: 'tomogram_group')
        opacity (float, optional):
            Opacity of the markers (default: 0.7)
        marker_size (int, optional):
            Size of the markers (default: 5)
        height (int, optional): 
            Height of the figure in pixels (default: 700)
        width (int, optional): 
            Width of the figure in pixels (default: 900)
        
    Returns:
        A Plotly 3D scatter plot figure showing distribution of motor positions in 3D space
        
    Example:
        >>> fig = visualize_motor_3d_distribution(labels_df, color_by='z_position')
        >>> fig.show(renderer='iframe')
    """
    # Filter out rows with -1 values (no motor present)
    filtered_df = df[(df['Motor axis 0'] >= 0) & 
                     (df['Motor axis 1'] >= 0) & 
                     (df['Motor axis 2'] >= 0)].copy()
    
    # Create figure
    fig = go.Figure()
    
    if color_by == 'tomogram_group':
        # Create groups based on number of motors (1, 2-3, 4+)
        filtered_df['group'] = pd.cut(
            filtered_df['Number of motors'], 
            bins=[0, 1, 3, float('inf')],
            labels=['Single Motor', '2-3 Motors', '4+ Motors']
        )
        
        # Define colors for each group
        colors = {
            'Single Motor': 'rgb(31,119,180)',  # Blue
            '2-3 Motors': 'rgb(255,127,14)',    # Orange
            '4+ Motors': 'rgb(214,39,40)'       # Red
        }
        
        # Plot each group separately
        for group, color in colors.items():
            group_df = filtered_df[filtered_df['group'] == group]
            
            if len(group_df) > 0:
                hover_text = []
                for idx, row in group_df.iterrows():
                    hover_text.append(
                        f"<b>Tomogram ID</b>: {row['tomo_id']}<br>"
                        f"<b>X Position</b>: {row['Motor axis 2']}<br>"
                        f"<b>Y Position</b>: {row['Motor axis 1']}<br>"
                        f"<b>Z Position</b>: {row['Motor axis 0']}<br>"
                        f"<b>Group</b>: {group}<br>"
                        f"<b>Motors in Tomogram</b>: {row['Number of motors']}"
                    )
                
                fig.add_trace(go.Scatter3d(
                    x=group_df['Motor axis 2'],
                    y=group_df['Motor axis 1'],
                    z=group_df['Motor axis 0'],
                    mode='markers',
                    marker=dict(
                        size=marker_size,
                        color=color,
                        opacity=opacity
                    ),
                    text=hover_text,
                    hovertemplate="%{text}<extra></extra>",
                    name=group,
                    showlegend=True
                ))
                
    else:
        # Prepare coloring based on selected attribute
        color_data = None
        colorscale = 'agsunset_r'
        colorbar_title = ""
        
        if color_by == 'z_position':
            color_data = filtered_df['Motor axis 0']
            colorbar_title = "<b>Z Position<br>(Slice Number)</b>"
            
        elif color_by == 'voxel_spacing':
            color_data = filtered_df['Voxel spacing']
            colorbar_title = "<b>Voxel Spacing<br>(Angstroms per Voxel)</b>"
            
        elif color_by == 'tomo_id':
            # Not recommended for many tomograms
            filtered_df['tomo_id_code'] = pd.Categorical(filtered_df['tomo_id']).codes
            color_data = filtered_df['tomo_id_code']
            colorbar_title = "<b>Tomogram ID</b>"
            
        elif color_by == 'motor_count':
            # Original coloring method
            color_data = filtered_df['Number of motors']
            colorbar_title = "<b>Number of Motors<br>in Tomogram</b>"
        
        else:
            # Default to z-position if invalid option
            color_data = filtered_df['Motor axis 0']
            colorbar_title = "<b>Z Position<br>(Slice Number)</b>"
        
        # Add specific hover text
        hover_text = []
        for idx, row in filtered_df.iterrows():
            hover_text.append(
                f"<b>Tomogram ID</b>: {row['tomo_id']}<br>"
                f"<b>X Position</b>: {row['Motor axis 2']}<br>"
                f"<b>Y Position</b>: {row['Motor axis 1']}<br>"
                f"<b>Z Position</b>: {row['Motor axis 0']}<br>"
                f"<b>Motors in Tomogram</b>: {row['Number of motors']}<br>"
                f"<b>Voxel Spacing</b>: {row['Voxel spacing']}"
            )
        
        # Create 3D scatter plot with continuous color scale
        fig.add_trace(go.Scatter3d(
            x=filtered_df['Motor axis 2'],
            y=filtered_df['Motor axis 1'],
            z=filtered_df['Motor axis 0'],
            mode='markers',
            marker=dict(
                size=marker_size,
                color=color_data,
                colorscale=colorscale,
                opacity=opacity,
                colorbar=dict(
                    title=colorbar_title,
                    thickness=20,
                    x=0.9
                )
            ),
            text=hover_text,
            hovertemplate="%{text}<extra></extra>",
            showlegend=False
        ))
    
    # Determine title based on coloring method
    title_text = '<b>3D Distribution of Motor Positions</b>'
    if color_by == 'tomogram_group':
        title_text += '<br><sup>Colored by groups: Single Motor, 2-3 Motors, 4+ Motors</sup>'
    elif color_by == 'z_position':
        title_text += '<br><sup>Colored by Z Position (Slice Number)</sup>'
    elif color_by == 'voxel_spacing':
        title_text += '<br><sup>Colored by Voxel Spacing (Resolution)</sup>'
    elif color_by == 'tomo_id':
        title_text += '<br><sup>Colored by Tomogram ID</sup>'
    elif color_by == 'motor_count':
        title_text += '<br><sup>Colored by Number of Motors in Tomogram</sup>'
    
    # Update layout for better readability
    fig.update_layout(
        title=dict(
            text=title_text,
            font=dict(size=22)
        ),
        scene=dict(
            xaxis_title='<b>X Position</b>',
            yaxis_title='<b>Y Position</b>',
            zaxis_title='<b>Z Position</b>',
            xaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray'),
            yaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray'),
            zaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray'),
        ),
        margin=dict(l=0, r=0, t=100, b=0),  # Increased top margin for subtitle
        title_x=0.5,
        title_y=0.97,
        height=height,
        width=width,
        font=dict(family="Arial, sans-serif"),
        legend=dict(
            title="<b>Tomogram Group</b>",
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255, 255, 255, 0.6)",
            bordercolor="gray",
            borderwidth=1,
            itemsizing='constant'
        )
    )
    
    return fig


# Default - Group by motor count (1, 2-3, 4+)
fig1 = visualize_motor_3d_distribution(labels_df, color_by="tomogram_group")
fig1.show(renderer='iframe')


# # Color by Z-position (depth/slice)
fig2 = visualize_motor_3d_distribution(labels_df, color_by='z_position')
fig2.show(renderer='iframe')


# Color by tomogram resolution
fig3 = visualize_motor_3d_distribution(labels_df, color_by='voxel_spacing')
fig3.show(renderer='iframe')


def visualize_motor_positions_by_tomogram(
    df: pd.DataFrame,
    tomogram_id_list: list[str] | None = None,
    n_top: int = 5,
    height: int = 500,
    width: int = 800
) -> dict[str, go.Figure]:
    """Creates a set of visualizations showing motor positions within the top N tomograms.

    We default to using the tomograms with the most motors to make things easier if no list is provided.
    
    Args:
        df (pd.DataFrame): 
            The tomogram dataset with at least 'tomo_id', 'Motor axis 0/1/2' columns
        tomogram_id_list (list[str], optional):
            Optional list of tomograms to return as plottable figures in dictionary
        n_top (int, optional):
            Number of top tomograms to visualize (default: 10)
        height (int, optional): 
            Height of each figure in pixels (default: 500)
        width (int, optional): 
            Width of each figure in pixels (default: 800)
        
    Returns:
        A dictionary of Plotly figures showing motor positions within each tomogram
        
    Example:
        >>> figs = visualize_motor_positions_by_tomogram(labels_df, n_top=5)
        >>> for tomo_id, fig in figs.items():
        >>>     fig.show(renderer='iframe')
    """
    if not tomogram_id_list:
        # Filter out rows with -1 values (no motor present)
        filtered_df = df[(df['Motor axis 0'] >= 0) & 
                         (df['Motor axis 1'] >= 0) & 
                         (df['Motor axis 2'] >= 0)]
    else:
        filtered_df = df[df['tomo_id'].isin(tomogram_id_list)]
    
    # Get top N tomograms with most motors
    top_tomos = filtered_df['tomo_id'].value_counts().head(n_top).index.tolist()
    
    # Create a figure for each top tomogram
    figures = {}
    
    for tomo_id in top_tomos:
        tomo_df = filtered_df[filtered_df['tomo_id'] == tomo_id]
        
        # Get the dimensions of this tomogram
        tomo_shape = (
            tomo_df['Array shape (axis 0)'].iloc[0],
            tomo_df['Array shape (axis 1)'].iloc[0],
            tomo_df['Array shape (axis 2)'].iloc[0]
        )
        
        motor_count = tomo_df['Number of motors'].iloc[0]
        if motor_count==0:
            print(f"\n... [SKIPPING] No Motors Found For tomo_id={tomo_id} [SKIPPING] ...\n")
            continue
            
        # Create 3D scatter plot for this tomogram
        fig = go.Figure(data=[go.Scatter3d(
            x=tomo_df['Motor axis 2'],
            y=tomo_df['Motor axis 1'],
            z=tomo_df['Motor axis 0'],
            mode='markers',
            marker=dict(
                size=10,
                color='red',
                symbol='circle',  # Valid symbol for Scatter3d
                opacity=0.8
            ),
            hovertemplate="<b>Motor Position</b><br>" +
                          "X: %{x}<br>" +
                          "Y: %{y}<br>" +
                          "Z: %{z}<extra></extra>",
            name="Motors"
        )])
        
        # Create wireframe box to represent tomogram boundaries
        fig = add_wireframe_box(
            fig, 
            x0=0, y0=0, z0=0, 
            x1=tomo_shape[2], y1=tomo_shape[1], z1=tomo_shape[0]
        )
        
        # Update layout for better readability
        fig.update_layout(
            title=dict(
                text=f'<b>Motor Positions in {tomo_id}</b><br><sup>Total Motors: {motor_count}</sup>',
                font=dict(size=18)
            ),
            scene=dict(
                xaxis_title='<b>X Position</b>',
                yaxis_title='<b>Y Position</b>',
                zaxis_title='<b>Z Position</b>',
                aspectmode='data',  # Preserve the shape proportions
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)  # Adjust camera position for better view
                )
            ),
            margin=dict(l=0, r=0, t=80, b=0),
            title_x=0.5,
            title_y=0.97,
            height=height,
            width=width,
            font=dict(family="Arial, sans-serif"),
            showlegend=True,
            legend=dict(
                title="<b>Components</b>",
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(255, 255, 255, 0.6)",
                bordercolor="gray",
                borderwidth=1
            )
        )
        
        figures[tomo_id] = fig
    
    return figures


def add_wireframe_box(fig: go.Figure, x0: int, y0: int, z0: int, x1: int, y1: int, z1: int) -> go.Figure:
    """Adds a wireframe box to a 3D figure to represent tomogram boundaries.
    
    Args:
        fig (go.Figure): Plotly figure object
        x0 (int): Minimum coordinates (x position)
        y0 (int): Minimum coordinates (y position)
        z0 (int): Minimum coordinates (z position)
        x1 (int): Maximum coordinates (x position)
        y1 (int): Maximum coordinates (y position)
        z1 (int): Maximum coordinates (z position)
        
    Returns:
        go.Figure:
            Updated Plotly figure with wireframe box
    """
    # Create the 8 corners of the box
    x = [x0, x1, x1, x0, x0, x1, x1, x0]
    y = [y0, y0, y1, y1, y0, y0, y1, y1]
    z = [z0, z0, z0, z0, z1, z1, z1, z1]
    
    # Define the 12 lines connecting the corners
    lines = [
        # Bottom face
        [0, 1], [1, 2], [2, 3], [3, 0],
        # Top face
        [4, 5], [5, 6], [6, 7], [7, 4],
        # Connecting edges
        [0, 4], [1, 5], [2, 6], [3, 7]
    ]
    
    # Add each line as a separate trace
    for line in lines:
        fig.add_trace(go.Scatter3d(
            x=[x[line[0]], x[line[1]]],
            y=[y[line[0]], y[line[1]]],
            z=[z[line[0]], z[line[1]]],
            mode='lines',
            line=dict(color='blue', width=2),
            hoverinfo='none',
            showlegend=False
        ))
    
    # Add a helper trace for the legend
    fig.add_trace(go.Scatter3d(
        x=[None], y=[None], z=[None],
        mode='lines',
        line=dict(color='blue', width=2),
        name='Tomogram Boundary',
        showlegend=True
    ))
    
    return fig


# Generate visualizations for top 5 tomograms with most motors
# tomo_figs = visualize_motor_positions_by_tomogram(labels_df, n_top=5)
tomo_figs = visualize_motor_positions_by_tomogram(labels_df, tomogram_id_list=['tomo_226cd8', 'tomo_003acc'])

# Display each figure individually
for tomo_id, fig in tomo_figs.items():
    rich.print(f"\n\n[bold red]Displaying visualization for tomogram: {tomo_id}[/bold red]\n")
    fig.show(renderer='iframe')


# def visualize_tomogram_dimensions(
#     df: pd.DataFrame,
#     height: int = 500,
#     width: int = 800
# ) -> go.Figure:
#     """
#     Creates a 3D scatter plot visualizing tomogram dimensions with each point 
#     representing a unique tomogram.
    
#     Args:
#         df (pd.DataFrame): 
#             The tomogram dataset with 'Array shape' columns
#         height (int, optional): 
#             Height of the figure in pixels (default: 500)
#         width (int, optional): 
#             Width of the figure in pixels (default: 800)
        
#     Returns:
#         A Plotly 3D scatter plot showing tomogram dimensions
        
#     Example:
#         >>> fig = visualize_tomogram_dimensions(labels_df)
#         >>> fig.show(renderer='iframe')
#     """
#     # Get unique tomograms
#     unique_tomos = df.drop_duplicates(subset=['tomo_id'])
    
#     # Create hover text
#     hover_text = []
#     for _, row in unique_tomos.iterrows():
#         hover_text.append(
#             f"<b>Tomogram ID</b>: {row['tomo_id']}<br>" +
#             f"<b>Z Dimension</b>: {row['Array shape (axis 0)']}<br>" +
#             f"<b>Y Dimension</b>: {row['Array shape (axis 1)']}<br>" +
#             f"<b>X Dimension</b>: {row['Array shape (axis 2)']}<br>" +
#             f"<b>Voxel Spacing</b>: {row['Voxel spacing']}<br>" +
#             f"<b>Number of Motors</b>: {row['Number of motors']}"
#         )
    
#     # Create 3D scatter plot
#     fig = go.Figure(data=[go.Scatter3d(
#         x=unique_tomos['Array shape (axis 2)'],  # X dimension
#         y=unique_tomos['Array shape (axis 1)'],  # Y dimension
#         z=unique_tomos['Array shape (axis 0)'],  # Z dimension
#         mode='markers',
#         marker=dict(
#             size=8,
#             color=unique_tomos['Number of motors'],
#             colorscale='Viridis',
#             opacity=0.8,
#             colorbar=dict(
#                 title="<b>Number of Motors</b>",
#                 thickness=20,
#             )
#         ),
#         text=hover_text,
#         hovertemplate="%{text}<extra></extra>"
#     )])
    
#     # Update layout
#     fig.update_layout(
#         title=dict(
#             text='<b>Tomogram Dimensions</b><br><sup>Each point represents one tomogram</sup>',
#             font=dict(size=22)
#         ),
#         scene=dict(
#             xaxis_title='<b>X Dimension (pixels)</b>',
#             yaxis_title='<b>Y Dimension (pixels)</b>',
#             zaxis_title='<b>Z Dimension (slices)</b>',
#             xaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray'),
#             yaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray'),
#             zaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray'),
#         ),
#         margin=dict(l=0, r=0, t=100, b=0),
#         title_x=0.5,
#         title_y=0.97,
#         height=height,
#         width=width,
#         font=dict(family="Arial, sans-serif")
#     )
    
#     return fig

# # Visualize tomogram dimensions in 3D space
# dim_fig = visualize_tomogram_dimensions(labels_df)
# dim_fig.show(renderer='iframe')


def visualize_tomogram_dimension_distribution(
    df: pd.DataFrame,
    axis: int,
    color: str = '#d27582',
    bin_width: int | None = None,
    height: int = 500,
    width: int = 800
) -> go.Figure:
    """Creates a histogram visualization showing the distribution of tomogram dimensions.
    
    Visualizes distribution along a specified axis (0: z, 1: y, 2: x).
    
    Args:
        df (pd.DataFrame): 
            The tomogram dataset with 'Array shape' columns
        axis (int): 
            The axis to visualize (0 for z, 1 for y, 2 for x)
        color (str, optional):
            Color for the histogram bars (default: '#2c7fb8')
        bin_width (int, optional):
            Width of histogram bins. If None, automatically determined (default: None)
        height (int, optional): 
            Height of the figure in pixels (default: 500)
        width (int, optional): 
            Width of the figure in pixels (default: 800)
        
    Returns:
        A Plotly histogram figure showing distribution of tomogram dimensions along specified axis
        
    Example:
        >>> fig = visualize_tomogram_dimension_distribution(labels_df, axis=0)
        >>> fig.show(renderer='iframe')
    """
    # Get unique tomograms
    unique_tomos = df.drop_duplicates(subset=['tomo_id'])
    
    # Define axis labels and column names
    axis_names = {0: 'Z (Slices)', 1: 'Y (Height)', 2: 'X (Width)'}
    axis_label = axis_names[axis]
    column_name = f'Array shape (axis {axis})'
    
    # Determine bin width if not specified
    if bin_width is None:
        range_of_values = unique_tomos[column_name].max() - unique_tomos[column_name].min()
        bin_width = max(1, round(range_of_values / 25))  # Default to 25 bins, minimum width of 1
    
    # Create histogram
    fig = px.histogram(
        unique_tomos, 
        x=column_name,
        nbins=None,  # Let Plotly determine bins based on bin_width
        histnorm=None,  # Count values directly
        color_discrete_sequence=[color],
        title=f'<b>Distribution of Tomogram Dimensions - {axis_label}</b>',
        labels={column_name: f'{axis_label} Dimension (pixels)'},
        height=height,
        width=width
    )
    
    # Update layout for better readability
    fig.update_layout(
        xaxis_title=f'<b>{axis_label} Dimension (pixels)</b>',
        yaxis_title='<b>Count</b>',
        margin=dict(l=40, r=40, t=80, b=40),
        title=dict(
            font=dict(size=22),  # Larger title
        ),
        title_x=0.5,  # Center the title
        title_y=0.95,  # Position title higher
        font=dict(family="Arial, sans-serif"),
        bargap=0.1  # Gap between bars
    )
    
    # Customize axis appearance
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgray',
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor='black',
        tickfont=dict(size=12)
    )
    
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgray',
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor='black',
        tickfont=dict(size=12)
    )
    
    # Add mean line
    mean_dimension = unique_tomos[column_name].mean()
    fig.add_vline(
        x=mean_dimension, 
        line_dash="dash", 
        line_color="yellow",
        annotation_text=f"Mean: {mean_dimension:.1f}",
        annotation_position="top right"
    )
    
    # Add median line
    median_dimension = unique_tomos[column_name].median()
    fig.add_vline(
        x=median_dimension, 
        line_dash="dot", 
        line_color="blue",
        annotation_text=f"Median: {median_dimension:.1f}",
        annotation_position="top left"
    )
    
    # Enhance hover information
    fig.update_traces(
        hovertemplate=f'<b>{axis_label} Dimension</b>: %{{x}}<br><b>Count</b>: %{{y}}<extra></extra>'
    )
    
    return fig


# Visualize the Z-axis (slice) distribution
z_dist_fig = visualize_tomogram_dimension_distribution(labels_df, axis=0)
z_dist_fig.show(renderer='iframe')

# Visualize the Y-axis distribution
y_dist_fig = visualize_tomogram_dimension_distribution(labels_df, axis=1)
y_dist_fig.show(renderer='iframe')

# Visualize the X-axis distribution
x_dist_fig = visualize_tomogram_dimension_distribution(labels_df, axis=2)
x_dist_fig.show(renderer='iframe')


def visualize_tomogram_shape_profiles(
    df: pd.DataFrame,
    n_profiles: int = 10,
    height: int = 600,
    width: int = 900
) -> go.Figure:
    """Creates a visualization showing dimension profiles of tomograms.
    
    This displays the relative dimensions across all three axes for the most common
    tomogram shapes in the dataset.
    
    Args:
        df (pd.DataFrame): 
            The tomogram dataset with 'Array shape' columns
        n_profiles (int, optional):
            Number of most common tomogram profiles to show (default: 10)
        height (int, optional): 
            Height of the figure in pixels (default: 600)
        width (int, optional): 
            Width of the figure in pixels (default: 900)
        
    Returns:
        A Plotly figure showing the most common tomogram shape profiles
        
    Example:
        >>> fig = visualize_tomogram_shape_profiles(labels_df)
        >>> fig.show(renderer='iframe'
    """
    # Get unique tomograms
    unique_tomos = df.drop_duplicates(subset=['tomo_id']).copy()
    
    # Create a shape profile string for each tomogram
    unique_tomos['shape_profile'] = unique_tomos.apply(
        lambda row: f"{int(row['Array shape (axis 0)'])}Ã—{int(row['Array shape (axis 1)'])}Ã—{int(row['Array shape (axis 2)'])}",
        axis=1
    )
    
    # Get the most common profiles
    top_profiles = unique_tomos['shape_profile'].value_counts().head(n_profiles)
    profile_counts = top_profiles.reset_index()
    profile_counts.columns = ['shape_profile', 'count']
    
    # Prepare data for radar/polar chart
    shapes = []
    for profile in top_profiles.index:
        # Extract dimensions from profile string
        dims = [int(d) for d in profile.split('Ã—')]
        
        # Add to shapes list
        shapes.append({
            'shape_profile': profile,
            'count': top_profiles[profile],
            'Z': dims[0],
            'Y': dims[1],
            'X': dims[2]
        })
    
    # Create DataFrame for plotting
    shapes_df = pd.DataFrame(shapes)
    
    # Normalize dimensions for radar chart
    for axis in ['Z', 'Y', 'X']:
        max_val = shapes_df[axis].max()
        shapes_df[f'{axis}_norm'] = shapes_df[axis] / max_val
    
    # Create figure
    fig = go.Figure()
    
    # Add a trace for each profile
    for i, row in shapes_df.iterrows():
        fig.add_trace(go.Scatterpolar(
            r=[row['Z_norm'], row['Y_norm'], row['X_norm'], row['Z_norm']],  # Close the loop
            theta=['Z', 'Y', 'X', 'Z'],  # Close the loop
            fill='toself',
            name=f"{row['shape_profile']} (n={row['count']})",
            hoverinfo='text',
            hovertext=(
                f"<b>Profile</b>: {row['shape_profile']}<br>"
                f"<b>Count</b>: {row['count']}<br>"
                f"<b>Z</b>: {row['Z']}<br>"
                f"<b>Y</b>: {row['Y']}<br>"
                f"<b>X</b>: {row['X']}"
            )
        ))
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f'<b>Top {n_profiles} Tomogram Shape Profiles</b><br><sup>Normalized dimensions</sup>',
            font=dict(size=22)
        ),
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=True,
        legend=dict(
            title="<b>Shape Profiles</b>",
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99,
            bgcolor="rgba(255, 255, 255, 0.6)",
            bordercolor="gray",
            borderwidth=1
        ),
        margin=dict(l=80, r=80, t=100, b=80),
        title_x=0.5,
        title_y=0.95,
        height=height,
        width=width,
        font=dict(family="Arial, sans-serif")
    )
    
    return fig


shape_profiles_fig = visualize_tomogram_shape_profiles(labels_df)
shape_profiles_fig.show(renderer='iframe')


def visualize_slice_with_motors(
    tomo_id: str,
    box_size: int = 40,
    figsize: tuple = (20, 20),
    slice_idx: int | None = None,
    df: pd.DataFrame | None = None,
    train_dir: str | None = None,
    show_titlebox: bool = False
) -> tuple:
    """Visualizes a tomogram slice with motors highlighted using colored transparent patches.
    
    This function creates a visualization of a single tomogram slice with annotated
    motors using semi-transparent colored boxes. Each motor gets a distinct color
    to improve visibility of small structures.
    
    Args:
        tomo_id (str): String identifier for the tomogram.
        box_size (int, optional): Integer size of bounding box to draw around motors (default: 24).
        figsize (tuple[int], optional): Tuple of (width, height) for the output figure (default: (12, 10)).
        slice_idx (int, optional): Integer Z-axis slice index to visualize. If unset, we will use the first slice with a motor.
        df (pd.Dataframe, optional): Pandas DataFrame containing motor coordinates and metadata.
        train_dir (str, optional): String path to directory containing tomogram data.
        show_titlebox (bool, optional): Display optional bounding box label and instance count
        
    Returns:
        tuple: (fig, ax) Matplotlib figure and axes objects for further customization.
        
    Raises:
        FileNotFoundError: If the specified slice image cannot be found.
    """
    # (0) Set defaults if not already set
    df = df or labels_df
    train_dir = train_dir or TRAIN_DIR
    
    # If no slice_idx is provided, use the first slice with a motor
    if slice_idx is None:
        motors_in_tomo = df[df['tomo_id'] == tomo_id]
        if len(motors_in_tomo) == 0:
            print(f"Error: No motors found for tomogram {tomo_id}")
            return None, None
        slice_idx = int(motors_in_tomo["Motor axis 0"].values[0])
    
    # (1) Construct the path to the slice image file
    slice_path = os.path.join(train_dir, tomo_id, f"slice_{slice_idx:04d}.jpg")
    
    # (2) Verify the slice exists and load it
    if not os.path.exists(slice_path):
        print(f"Error: Slice {slice_path} does not exist")
        return None, None
    
    # (3) Load the image data as a numpy array
    img = np.array(Image.open(slice_path))
    
    # (4) Filter the dataframe to get motors for this specific tomogram and slice
    motors = df[(df['tomo_id'] == tomo_id) & (df['Motor axis 0'] == slice_idx)]
    
    # (5) Create a new figure and axis for visualization
    fig, ax = plt.subplots(figsize=figsize)
    
    # (6) Display the grayscale tomogram slice
    ax.imshow(img, cmap='gray')
    
    # (7) Define a carefully selected color palette for motor annotations
    # These colors are chosen to be visually distinct but harmonious
    motor_colors = [
        '#1f77b4',  # Blue
        '#ff7f0e',  # Orange
        '#2ca02c',  # Green
        '#d62728',  # Red
        '#9467bd',  # Purple
        '#8c564b',  # Brown
        '#e377c2',  # Pink
        '#7f7f7f',  # Gray
        '#bcbd22',  # Olive
        '#17becf'   # Cyan
    ]
    
    # (8) Annotate each motor in the current slice
    for i, (_, motor) in enumerate(motors.iterrows()):
        # (8a) Select a color for this motor, cycling through the palette if needed
        color_idx = i % len(motor_colors)
        motor_color = motor_colors[color_idx]
        
        # (8b) Extract the motor coordinates (integer pixel positions)
        y = int(motor['Motor axis 1'])
        x = int(motor['Motor axis 2'])
        half_box = box_size // 2
        
        # (8c) Create a semi-transparent rectangle to highlight the motor
        # Thin border with matching fill color for optimal visibility
        rect = Rectangle(
            (x - half_box, y - half_box),  # Upper-left corner position
            box_size, box_size,            # Width and height
            linewidth=3.0,                 # Border
            edgecolor=motor_color,         # Border color
            facecolor=motor_color,         # Fill with same color as border
            alpha=0.3                      # Semi-transparent fill for visibility
        )
        ax.add_patch(rect)
        
        # (8d) Add a label identifying the motor with good contrast
        if show_titlebox:
            ax.text(
                x, y - half_box - 5,           # Position just above the box
                f"Motor {i+1}",                # Label text with motor number
                color='white',                 # White text for readability
                fontsize=9,                    # Readable but not oversized font
                fontweight='bold',             # Bold for visibility against background
                bbox=dict(                     # Background box for contrast
                    facecolor=motor_color,     # Same color as the motor annotation
                    alpha=0.8,                 # Mostly opaque for readability
                    pad=1,                     # Small padding around text
                    boxstyle='round,pad=0.3'   # Slightly rounded corners
                ),
                ha='center',                   # Center-align text horizontally
                zorder=10                      # Ensure text appears above other elements
            )
    
    # (9) Add informative title and metadata
    ax.set_title(f"Tomogram: {tomo_id}, Slice: {slice_idx}", fontsize=14, fontweight='bold')
    
    # (10) Add a count of motors in the current slice
    ax.text(
        10, 20,                            # Position in upper-left corner
        f"Total Motors: {len(motors)}",    # Display count of motors
        color='white',                     # White text for visibility
        bbox=dict(                         # Background box
            facecolor='black',             # Black background
            alpha=0.5,                     # Partially transparent
            boxstyle='round,pad=0.5'       # Rounded corners
        ), 
        fontsize=10
    )
    
    # (11) Add a legend for the motor identifiers if motors are present
    if len(motors) > 0:
        # (11a) Create legend elements for each motor
        legend_elements = []
        for i in range(min(len(motors), len(motor_colors))):
            color = motor_colors[i % len(motor_colors)]
            legend_elements.append(
                plt.Line2D(
                    [0], [0],                  # Dummy coordinates
                    marker='s',                # Square marker matching annotations
                    color='w',                 # White edge
                    markerfacecolor=color,     # Fill with motor color
                    markersize=8,              # Visible but not too large
                    label=f'Motor {i+1}'       # Label with motor number
                )
            )
        
        # (11b) Place the legend in the upper right corner
        ax.legend(
            handles=legend_elements,
            loc='upper right',
            title='Motors',
            framealpha=0.7,                # Semi-transparent background
            fontsize='small',
            title_fontsize='small'
        )
    
    # (12) Hide axis for a cleaner visualization
    ax.axis('off')
    
    # (13) Ensure layout is clean and tight
    plt.tight_layout()
    
    # (14) Return the figure and axes for potential further customization
    return fig, ax


def visualize_multiple_slices(
    tomo_id: str,
    start_slice: int | None = None,
    end_slice: int | None = None, 
    step: int = 1,
    box_size: int = 30,
    figsize_x: int = 20,
    df: pd.DataFrame | None = None,
    train_dir: str | None = None,
    show_titlebox: bool = False
) -> tuple:
    """Visualizes multiple tomogram slices with motors highlighted.
    
    Creates a grid of visualizations showing multiple consecutive slices from a 
    tomogram to help understand the 3D distribution of motors. Consistent colors
    are used across slices for better tracking of structures. Only shows slices
    that contain motors.
    
    Args:
        tomo_id (str): String identifier for the tomogram.
        start_slice (int, optional): Integer starting Z-axis slice index. If None, uses first slice with a motor.
        end_slice (int, optional): Integer ending Z-axis slice index. If None, uses last slice with a motor.
        step (int, optional): Integer step size between visualized slices. Default is 1 (show all motor slices).
        box_size (int, optional): Integer size of bounding box to draw around motors (default: 34).
        figsize_x (int, optional): X dimension for output figure
        df (pd.Dataframe, optional): Pandas DataFrame containing motor coordinates and metadata.
        train_dir (str, optional): String path to directory containing tomogram data.
        show_titlebox (bool, optional): Display optional bounding box label and instance count
        
    Returns:
        tuple: (fig, axes) Matplotlib figure and axes objects for further customization.
    """
    # (0) Set defaults if not already set
    # Use the global labels_df if no dataframe is provided
    df = df or labels_df
    # Use the global TRAIN_DIR if no directory is provided
    train_dir = train_dir or TRAIN_DIR
    
    # (1) Get all slice indices that contain motors for this tomogram
    # First, filter the dataframe to only include motors in the requested tomogram
    motors_in_tomo = df[df['tomo_id'] == tomo_id]
    
    # Check if any motors exist for this tomogram
    if len(motors_in_tomo) == 0:
        print(f"Error: No motors found for tomogram {tomo_id}")
        return None, None
    
    # Extract unique slice indices that contain motors, convert to int, and sort
    # This ensures we only show slices that actually have motors
    all_motor_slices = sorted(motors_in_tomo["Motor axis 0"].astype(int).unique().tolist())
    
    # (2) Apply start_slice and end_slice filters if provided
    # If start_slice is specified, only include slices at or after that index
    if start_slice is not None:
        all_motor_slices = [s for s in all_motor_slices if s >= start_slice]
    
    # If end_slice is specified, only include slices at or before that index
    if end_slice is not None:
        all_motor_slices = [s for s in all_motor_slices if s <= end_slice]
    
    # (3) Apply step to select slices
    # Using Python's slice notation to take every 'step' slice
    # For example, step=2 will show every other slice
    slices = all_motor_slices[::step]
    
    # After applying all filters, verify we still have slices to display
    if len(slices) == 0:
        print(f"Error: No motor slices found for tomogram {tomo_id} with the given parameters")
        return None, None
    
    # Store the number of slices for later use in grid calculations
    n_slices = len(slices)
    
    # (4) Calculate optimal grid dimensions for the subplots
    # Limit to 3 columns maximum for readability
    cols = min(3, n_slices)  
    # Calculate how many rows are needed to fit all slices
    # Using integer division with ceiling to ensure all slices fit
    rows = (n_slices + cols - 1) // cols
    
    # (5) Create a figure with a grid of subplots
    # This creates a single figure with an array of axes (subplots)
    figsize_y = 8*rows
    fig, axes = plt.subplots(rows, cols, figsize=(figsize_x, figsize_y))
    
    # (6) Handle different axes array shapes based on grid dimensions
    # Matplotlib returns different structures depending on grid shape:
    if rows == 1 and cols == 1:
        # For a single subplot, convert to a 2D array for consistent indexing
        axes = np.array([[axes]])
    elif rows == 1:
        # For a single row, reshape to 2D array with shape (1, cols)
        axes = axes.reshape(1, -1)
    elif cols == 1:
        # For a single column, reshape to 2D array with shape (rows, 1)
        axes = axes.reshape(-1, 1)
    
    # (7) Define a consistent color palette for motor annotations
    # These colors are chosen to be distinct but visually harmonious
    # Using the Matplotlib default color cycle for consistency
    motor_colors = [
        '#1f77b4',  # Blue
        '#ff7f0e',  # Orange
        '#2ca02c',  # Green
        '#d62728',  # Red
        '#9467bd',  # Purple
        '#8c564b',  # Brown
        '#e377c2',  # Pink
        '#7f7f7f',  # Gray
        '#bcbd22',  # Olive
        '#17becf'   # Cyan
    ]
    
    # (8) Create a mapping of motor ID to color for consistency across slices
    # This crucial step ensures the same motor gets the same color in each slice,
    # making it easier to track structures across the Z-axis
    all_motors = motors_in_tomo
    unique_motors = {}  # Dictionary to map motor IDs to consistent colors
    color_counter = 0   # Counter to cycle through the color palette
    
    # (8a) Find min and max slice for this tomogram's motors
    # This helps establish the range for our motor tracking across slices
    min_slice = min(all_motor_slices)
    max_slice = max(all_motor_slices)
    
    # (8b) Group motors that are close to each other across slices
    # We scan a range that includes a 5-slice buffer on either side of our actual data
    # This helps track motors that might appear in slices we're not directly visualizing
    for z in range(min_slice - 5, max_slice + 6):  
        # Find all motors in the current slice
        slice_motors = all_motors[all_motors['Motor axis 0'] == z]
        
        # For each motor in this slice, create a unique identifier
        for _, motor in slice_motors.iterrows():
            # Create a unique ID based on y and x coordinates
            # Motors at similar positions in adjacent slices are likely the same structure
            motor_id = f"{int(motor['Motor axis 1'])}_{int(motor['Motor axis 2'])}"
            
            # If this is the first time we've seen this motor, assign it a color
            if motor_id not in unique_motors:
                unique_motors[motor_id] = motor_colors[color_counter % len(motor_colors)]
                color_counter += 1
    
    # (9) Process and visualize each slice
    for i, slice_idx in enumerate(slices):
        # (9a) Get the current subplot from our grid
        # Calculate row and column indices based on the current slice index
        row, col = i // cols, i % cols
        ax = axes[row, col]
        
        # (9b) Load the slice image
        # Construct the file path for the current slice JPEG
        slice_path = os.path.join(train_dir, tomo_id, f"slice_{slice_idx:04d}.jpg")
        
        # (9c) Handle missing slice files
        # If the image file doesn't exist, show an error message instead
        if not os.path.exists(slice_path):
            ax.text(0.5, 0.5, f"Slice {slice_idx} not found", 
                   ha='center', va='center', fontsize=10)
            ax.axis('off')
            continue
        
        # (9d) Load and display the image
        # Read the image file and convert to a numpy array
        img = np.array(Image.open(slice_path))
        # Display the grayscale image
        ax.imshow(img, cmap='gray')
        
        # (9e) Filter dataframe for motors in this specific slice
        # Get only the motors that match both the tomogram ID and the current slice
        motors = df[(df['tomo_id'] == tomo_id) & (df['Motor axis 0'] == slice_idx)]
        
        # (9f) Draw annotation boxes around each motor
        for j, (_, motor) in enumerate(motors.iterrows()):
            # Extract the integer pixel coordinates
            y = int(motor['Motor axis 1'])
            x = int(motor['Motor axis 2'])
            # Create a unique ID for this motor based on its position
            motor_id = f"{y}_{x}"
            
            # Choose a color for this motor
            # First try to use a consistent color if this motor has been seen before
            if motor_id in unique_motors:
                motor_color = unique_motors[motor_id]
            else:
                # Fallback to a sequence-based color if not mapped
                motor_color = motor_colors[j % len(motor_colors)]
            
            # Calculate the box dimensions
            half_box = box_size // 2
            
            # Create a semi-transparent rectangle to highlight the motor
            rect = Rectangle(
                (x - half_box, y - half_box),  # Upper-left corner position
                box_size, box_size,            # Width and height
                linewidth=2.0,                 # border
                edgecolor=motor_color,         # Border color
                facecolor=motor_color,         # Fill with same color as border
                alpha=0.3                      # Semi-transparent fill for visibility
            )
            # Add the rectangle to the plot
            ax.add_patch(rect)
            
            # Add a small label with the motor number
            #   - For the multi-slice view, we use a compact circular label in the center
            if show_titlebox:
                ax.text(
                    x, y,                          # Center position
                    f"{j+1}",                      # Simple numeric label
                    color='white',                 # White text for readability
                    fontsize=7,                    # Small font size to avoid overcrowding
                    fontweight='bold',             # Bold text for visibility
                    ha='center',                   # Center-align horizontally
                    va='center',                   # Center-align vertically
                    bbox=dict(                     # Background box for contrast
                        facecolor=motor_color,     # Use the same motor color
                        alpha=0.8,                 # Mostly opaque for readability
                        boxstyle='circle',         # Circular label shape
                        pad=0.1                    # Minimal padding to keep compact
                    ),
                    zorder=10                      # Ensure text appears above other elements
                )
        
        # (9g) Add slice information to the subplot title
        ax.set_title(f"Slice: {slice_idx}, Motors: {len(motors)}", fontsize=10)
        # Hide axis for a cleaner visualization
        ax.axis('off')
    
    # (10) Hide any empty subplots
    # If our grid has more cells than slices, hide the extras
    for i in range(n_slices, rows * cols):
        row, col = i // cols, i % cols
        axes[row, col].axis('off')
    
    # (11) Add an overall title for the figure
    # Show the slice range if multiple slices, otherwise just the single slice
    slice_range = f"Slices: {slices[0]}-{slices[-1]}" if len(slices) > 1 else f"Slice: {slices[0]}"
    plt.suptitle(f"Tomogram: {tomo_id} - {slice_range}", fontsize=16, fontweight='bold')
    
    # (12) Adjust layout for optimal viewing
    # Ensure subplots don't overlap
    plt.tight_layout()
    # Make room for the overall title at the top
    plt.subplots_adjust(top=0.98)
    
    # (13) Return the figure and axes for potential further customization
    return fig, axes


def normalize_and_enhance_contrast(img: np.ndarray, clip_percentile: float = 0.5) -> np.ndarray:
    """Normalizes and enhances contrast in tomogram slices for better visualization.
    
    Args:
        img (np.ndarray): Input image as numpy array.
        clip_percentile (float, optional): Percentile value for contrast clipping (default: 0.5).
        
    Returns:
        np.ndarray: Contrast-enhanced normalized image.
    """
    # (1) Determine value range for contrast enhancement
    p_low = np.percentile(img, clip_percentile)
    p_high = np.percentile(img, 100 - clip_percentile)
    
    # (2) Clip the image values to the determined range
    img_clipped = np.clip(img, p_low, p_high)
    
    # (3) Normalize to 0-1 range
    img_normalized = (img_clipped - p_low) / (p_high - p_low)
    
    return img_normalized


# visualize_slice_with_motors("tomo_226cd8")  # Shows first slice with a motor
# visualize_slice_with_motors("tomo_226cd8", slice_idx=169)  # Shows specific slice
# visualize_multiple_slices("tomo_226cd8")  # Shows all slices with motors
_f1, _ax1 = visualize_multiple_slices("tomo_00e463")
_f2, _ax2 = visualize_slice_with_motors("tomo_00e463", slice_idx=225)


def prepare_yolo_dataset(
    train_dir: str,
    train_labels_path: str,
    yolo_dataset_dir: str,
    yolo_images_train: str,
    yolo_images_val: str,
    yolo_labels_train: str,
    yolo_labels_val: str,
    trust: int = 4,
    box_size: int = 24,
    train_split: float = 0.8,
    random_seed: int = 42,
    return_labels_df: bool = True
) -> dict[str, Any]:
    """
    Prepare the complete YOLO dataset from tomograms.
    
    This is the main function that orchestrates the entire dataset preparation process.
    
    Args:
        train_dir (str): Directory containing the raw tomogram training data
        train_labels_path (str): Path to the CSV file with motor labels
        yolo_dataset_dir (str): Base directory for the YOLO dataset
        yolo_images_train (str): Directory for training images
        yolo_images_val (str): Directory for validation images
        yolo_labels_train (str): Directory for training labels
        yolo_labels_val (str): Directory for validation labels
        trust (int, optional): Number of slices above and below center slice to include
        box_size (int, optional): Size of bounding box in pixels for annotations
        train_split (float, optional): Fraction of data to use for training (0.0-1.0)
        random_seed (int, optional): Random seed for reproducibility
        return_labels_df (bool, optional): Whether to return the loaded labels dataframe or not.
        
    Returns:
        dict[str, Any]:
            Summary statistics and paths
        
    Raises:
        Various exceptions if processing fails
    """
    # (1) Validate parameters
    if not 0.0 < train_split < 1.0:
        raise ValueError("train_split must be between 0.0 and 1.0")
    
    if trust < 0:
        raise ValueError("trust must be a non-negative integer")
        
    if box_size <= 0:
        raise ValueError("box_size must be a positive integer")
    
    # (2) Create necessary directories
    create_dataset_directories(
        yolo_images_train, 
        yolo_images_val, 
        yolo_labels_train, 
        yolo_labels_val
    )
    
    # (3) Load and validate the labels
    labels_df = validate_labels_file(train_labels_path)
    
    # (4) Count total number of motors for reporting
    total_motors = labels_df['Number of motors'].sum()
    print(f"\nTOTAL NUMBER OF MOTORS IN THE DATASET: {total_motors}")
    
    # (5) Split tomograms into training and validation sets
    train_tomos, val_tomos = split_tomograms(
        labels_df, 
        train_split=train_split, 
        random_seed=random_seed
    )
    
    # (6) Process training tomograms
    train_slices, train_motors = process_tomogram_set(
        labels_df,
        train_tomos, 
        train_dir,
        yolo_images_train, 
        yolo_labels_train, 
        "training",
        trust=trust,
        box_size=box_size
    )
    
    # (7) Process validation tomograms
    val_slices, val_motors = process_tomogram_set(
        labels_df,
        val_tomos, 
        train_dir,
        yolo_images_val, 
        yolo_labels_val, 
        "validation",
        trust=trust,
        box_size=box_size
    )
    
    # (8) Create YAML configuration file for YOLO
    yaml_path = create_yaml_config(yolo_dataset_dir)
    
    # (9) Create and populate summary statistics
    stats = {
        "train_tomograms": len(train_tomos),
        "val_tomograms": len(val_tomos),
        "train_motors": train_motors,
        "val_motors": val_motors,
        "train_slices": train_slices,
        "val_slices": val_slices,
        "dataset_dir": yolo_dataset_dir,
        "yaml_path": yaml_path
    }
    
    # (10) Print summary information
    print(f"\nProcessing Summary:")
    print(f"- Train set: {stats['train_tomograms']} tomograms, {train_motors} motors, {train_slices} slices")
    print(f"- Validation set: {stats['val_tomograms']} tomograms, {val_motors} motors, {val_slices} slices")
    print(f"- Total: {stats['train_tomograms'] + stats['val_tomograms']} tomograms, "
          f"{train_motors + val_motors} motors, {train_slices + val_slices} slices")
    
    # (11) Return statistics dictionary
    if not return_labels_df:
        return stats

    # (12) Optionally return the statistics dictionary and the labels dataframe
    return stats, labels_df

# Run the preprocessing
summary, labels_df = prepare_yolo_dataset(
    train_dir=TRAIN_DIR,
    train_labels_path=TRAIN_LABELS_PATH,
    yolo_dataset_dir=YOLO_DATASET_DIR,
    yolo_images_train=YOLO_IMAGES_TRAIN,
    yolo_images_val=YOLO_IMAGES_VAL,
    yolo_labels_train=YOLO_LABELS_TRAIN,
    yolo_labels_val=YOLO_LABELS_VAL,
    trust=TRUST,
    box_size=BOX_SIZE,
    train_split=TRAIN_SPLIT,
    random_seed=42
)

print(f"\nPreprocessing Complete:")
print(f"  - Training data: {summary['train_tomograms']} tomograms, {summary['train_motors']} motors, {summary['train_slices']} slices")
print(f"  - Validation data: {summary['val_tomograms']} tomograms, {summary['val_motors']} motors, {summary['val_slices']} slices")
print(f"  - Dataset directory: {summary['dataset_dir']}")
print(f"  - YAML configuration: {summary['yaml_path']}")
print(f"\nReady for YOLO training!")

