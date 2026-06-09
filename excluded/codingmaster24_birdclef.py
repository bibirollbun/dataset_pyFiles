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


import ast
import folium
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import logging
import seaborn as sns
from collections import Counter
from folium.plugins import MarkerCluster
from pathlib import Path # Using pathlib for better path handling
from IPython.display import IFrame, display, HTML #added HTML import
from io import StringIO
import warnings
warnings.filterwarnings("ignore")


BASE_INPUT_PATH = Path("/kaggle/input/birdclef-2025")
TRAIN_CSV_PATH = BASE_INPUT_PATH / "train.csv"
TAXONOMY_CSV_PATH = BASE_INPUT_PATH / "taxonomy.csv"
SAMPLE_SUBMISSION_PATH = BASE_INPUT_PATH / "sample_submission.csv"
RECORDING_LOCATION_PATH = BASE_INPUT_PATH / "recording_location.txt" # Path defined, though not used in original logic beyond existence check
TRAIN_AUDIO_DIR = BASE_INPUT_PATH / "train_audio"
TRAIN_SOUNDSCAPES_DIR = BASE_INPUT_PATH / "train_soundscapes"
TEST_SOUNDSCAPES_DIR = BASE_INPUT_PATH / "test_soundscapes"
OUTPUT_DIR = Path("./") # Output directory for plots and files
OUTPUT_DIR.mkdir(exist_ok=True) # Create output directory if it doesn't exist


# Plotting settings
sns.set_theme(style="whitegrid") # Consistent theme
plt.rcParams['figure.figsize'] = (14, 7) # Default figure size
TOP_N_SPECIES_PLOT = 50 # Limit the number of species shown in bar plots


# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_data(csv_path):
    """Loads a CSV file into a pandas DataFrame with basic error handling."""
    try:
        df = pd.read_csv(csv_path)
        logging.info(f"Successfully loaded data from: {csv_path}. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        logging.error(f"Error: File not found at {csv_path}")
        return None
    except Exception as e:
        logging.error(f"Error loading {csv_path}: {e}")
        return None


def plot_distribution(data_series, title, xlabel, ylabel, kind='bar', top_n=None, rotation=90, figsize=(14, 7)):
    """Generates and saves a distribution plot for a pandas Series."""
    plt.figure(figsize=figsize)
    if kind == 'bar':
        counts = data_series.value_counts()
        if top_n:
            counts = counts.head(top_n)
            title += f" (Top {top_n})"
        counts.plot(kind='bar')
        plt.xticks(rotation=rotation)
    elif kind == 'hist':
        sns.histplot(data_series, kde=True, bins=30)
        plt.xticks(rotation=0) # Usually no rotation needed for histograms
    else:
        logging.warning(f"Unsupported plot kind: {kind}")
        return

    plt.title(title, fontsize=16)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.tight_layout() # Adjust layout to prevent labels overlapping
    # Create a filename-safe version of the title
    filename_title = "".join(c if c.isalnum() else "_" for c in title).lower()
    save_path = OUTPUT_DIR / f"{filename_title}_distribution.png"
    plt.savefig(save_path)
    logging.info(f"Saved plot: {save_path}")
    plt.show()


def parse_secondary_labels(label_str):
    """Safely parses the string representation of secondary labels."""
    if pd.isna(label_str) or label_str == "[]" or not label_str:
        return []
    try:
        # Using ast.literal_eval is safer than eval()
        parsed_list = ast.literal_eval(label_str)
        # Ensure it's actually a list of strings
        if isinstance(parsed_list, list) and all(isinstance(item, str) for item in parsed_list):
             return parsed_list
        else:
             logging.warning(f"Could not parse secondary label correctly, unexpected format: {label_str}")
             return [] # Return empty list if format is unexpected
    except (ValueError, SyntaxError, TypeError) as e:
        logging.warning(f"Could not parse secondary label string: '{label_str}'. Error: {e}")
        return [] # Return empty list on error


def count_files_in_dir(directory):
    """Counts all files recursively in a given directory."""
    try:
        return sum(len(files) for _, _, files in os.walk(directory))
    except FileNotFoundError:
        logging.error(f"Directory not found: {directory}")
        return 0


def plot_correlation_matrix(df, columns, title="Correlation Matrix"):
    """Plots a heatmap of the correlation matrix for specified columns."""
    if not columns:
        logging.warning("No columns specified for correlation matrix.")
        return
    corr_matrix = df[columns].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title(title, fontsize=16)
    plt.tight_layout()
    save_path = OUTPUT_DIR / f"{title.replace(' ', '_').lower()}_heatmap.png"
    plt.savefig(save_path)
    logging.info(f"Saved plot: {save_path}")
    plt.show()


def plot_geo_locations(df, lat_col='latitude', lon_col='longitude', popup_col='primary_label', map_filename="recording_locations_map.html"):
    """Creates and saves an interactive map of geographical locations."""
    location_data = df.dropna(subset=[lat_col, lon_col]).copy() # Work on a copy
    if location_data.empty:
        logging.warning("No valid location data found to plot map.")
        return

    # Filter out potential invalid coordinates (basic check)
    location_data = location_data[
        (location_data[lat_col].between(-90, 90)) &
        (location_data[lon_col].between(-180, 180))
    ]
    if location_data.empty:
        logging.warning("No valid location data after basic lat/lon range filtering.")
        return

    logging.info(f"Plotting {location_data.shape[0]} locations on map.")

    mean_lat = location_data[lat_col].mean()
    mean_lon = location_data[lon_col].mean()

    m = folium.Map(location=[mean_lat, mean_lon], zoom_start=5) # Adjusted zoom

    marker_cluster = MarkerCluster().add_to(m)

    for idx, row in location_data.iterrows():
        popup_text = f"Species: {row[popup_col]}<br>Rating: {row.get('rating', 'N/A')}" # Add rating if available
        folium.Marker(
            location=[row[lat_col], row[lon_col]],
            popup=popup_text,
            tooltip=f"Click for info on {row[popup_col]}" # Add tooltip
        ).add_to(marker_cluster)

    save_path = OUTPUT_DIR / map_filename
    m.save(save_path)
    logging.info(f"Interactive map saved as: {save_path}")


if __name__ == "__main__":

    logging.info("Starting BirdCLEF 2025 EDA Script")

    # --- Load Data ---
    train_df = load_data(TRAIN_CSV_PATH)
    taxonomy_df = load_data(TAXONOMY_CSV_PATH)
    sample_submission = load_data(SAMPLE_SUBMISSION_PATH)

    # Check if dataframes loaded successfully
    if train_df is None or taxonomy_df is None or sample_submission is None:
        logging.error("Failed to load one or more essential data files. Exiting.")
        exit()

    # --- Basic Data Inspection ---
    logging.info("--- Basic Data Inspection ---")
    logging.info(f"Train data info:\n{train_df.info()}")
    logging.info(f"Missing values in train_df:\n{train_df.isnull().sum()}")
    logging.info(f"Duplicates in train_df: {train_df.duplicated().sum()}")

    # --- Analyze Primary Labels ---
    logging.info("--- Analyzing Primary Labels ---")
    plot_distribution(train_df['primary_label'],
                      title="Distribution of Primary Species Labels",
                      xlabel="Species",
                      ylabel="Count",
                      kind='bar',
                      top_n=TOP_N_SPECIES_PLOT) # Plot only top N for clarity
    species_counts = train_df['primary_label'].value_counts()
    logging.info(f"Primary label count statistics:\n{species_counts.describe()}")
    logging.info(f"Most common primary labels:\n{species_counts.head()}")
    logging.info(f"Least common primary labels:\n{species_counts.tail()}")

    # --- Analyze Ratings ---
    logging.info("--- Analyzing Ratings ---")
    plot_distribution(train_df['rating'],
                      title="Distribution of Audio Quality Ratings",
                      xlabel="Rating",
                      ylabel="Frequency",
                      kind='hist')

    # Rating statistics and outlier detection
    logging.info(f"Rating Summary Statistics:\n{train_df['rating'].describe()}")
    plt.figure(figsize=(10, 4))
    sns.boxplot(x=train_df['rating'])
    plt.title("Boxplot of Ratings")
    plt.xlabel("Rating")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "rating_boxplot.png")
    plt.show()

    Q1 = train_df['rating'].quantile(0.25)
    Q3 = train_df['rating'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    logging.info(f"Rating outlier bounds (IQR method): Lower = {lower_bound:.2f}, Upper = {upper_bound:.2f}")
    rating_outliers = train_df[(train_df['rating'] < lower_bound) | (train_df['rating'] > upper_bound)]
    logging.info(f"Number of potential rating outliers: {rating_outliers.shape[0]}")
    if not rating_outliers.empty:
        logging.info(f"Sample rating outliers:\n{rating_outliers[['primary_label', 'rating']].head()}")

    # Average rating by species
    avg_rating_by_species = train_df.groupby('primary_label')['rating'].mean().sort_values(ascending=False)
    plt.figure(figsize=(14, 7))
    avg_rating_by_species.head(TOP_N_SPECIES_PLOT).plot(kind='bar') # Plot top N
    plt.title(f"Average Rating by Species (Top {TOP_N_SPECIES_PLOT})")
    plt.xlabel("Species")
    plt.ylabel("Average Rating")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "average_rating_by_species.png")
    plt.show()


    # --- Analyze Recording Types ---
    logging.info("--- Analyzing Recording Types ---")
    plot_distribution(train_df['type'],
                      title="Distribution of Recording Types",
                      xlabel="Type",
                      ylabel="Count",
                      kind='bar',
                      rotation=0) # Horizontal labels likely okay

    # --- Analyze Secondary Labels ---
    logging.info("--- Analyzing Secondary Labels ---")
    train_df['secondary_labels_list'] = train_df['secondary_labels'].apply(parse_secondary_labels)
    train_df['num_secondary_labels'] = train_df['secondary_labels_list'].apply(len)

    plot_distribution(train_df['num_secondary_labels'],
                      title="Distribution of Number of Secondary Labels per Recording",
                      xlabel="Number of Secondary Labels",
                      ylabel="Frequency",
                      kind='hist',
                      rotation=0)

    # Flatten the list of all secondary labels
    all_secondary_labels = [label for sublist in train_df['secondary_labels_list'] for label in sublist]
    secondary_counter = Counter(all_secondary_labels)
    logging.info(f"Total unique secondary labels: {len(secondary_counter)}")
    logging.info(f"Most common secondary labels (Top 15):\n{secondary_counter.most_common(15)}")

    # Plot top N secondary labels
    if secondary_counter:
        sec_labels_df = pd.DataFrame(secondary_counter.most_common(TOP_N_SPECIES_PLOT), columns=['label', 'count'])
        plt.figure(figsize=(14, 7))
        sns.barplot(x='count', y='label', data=sec_labels_df, palette='viridis')
        plt.title(f'Top {TOP_N_SPECIES_PLOT} Most Common Secondary Labels')
        plt.xlabel('Count')
        plt.ylabel('Species Label')
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "top_secondary_labels.png")
        plt.show()
    else:
        logging.info("No secondary labels found to plot.")

    # --- Analyze Taxonomy Data ---
    logging.info("--- Analyzing Taxonomy Data ---")
    logging.info(f"Taxonomy data info:\n{taxonomy_df.info()}")
    logging.info(f"Actual columns in taxonomy_df: {taxonomy_df.columns.tolist()}")
    plot_distribution(taxonomy_df['class_name'],
                      title="Distribution of Taxonomic Classes",
                      xlabel="Class",
                      ylabel="Count",
                      kind='bar',
                      rotation=0)

    # Distribution of Orders (more specific than class)
    plot_distribution(taxonomy_df['primary_label'],
                      title="Distribution of Taxonomic Orders",
                      xlabel="Order",
                      ylabel="Count",
                      kind='bar',
                      top_n=30, # Limit for readability
                      rotation=45)


    # --- Merge Train and Taxonomy Data ---
    # logging.info("--- Merging Train and Taxonomy Data ---")
    # merged_df = pd.merge(train_df, taxonomy_df, on="primary_label", how="left")
    # logging.info(f"Merged data shape: {merged_df.shape}")
    # missing_tax_info = merged_df['species_code'].isnull().sum() # Check based on a core taxonomy column
    # logging.info(f"Recordings with missing taxonomy info after merge: {missing_tax_info}")
    # if missing_tax_info > 0:
        # missing_labels = train_df[merged_df['species_code'].isnull()]['primary_label'].unique()
        # logging.warning(f"Primary labels in train_df missing in taxonomy_df: {list(missing_labels)}")

    # --- Analyze File Counts ---
    logging.info("--- Analyzing File Counts ---")
    logging.info(f"Train Audio files count: {count_files_in_dir(TRAIN_AUDIO_DIR)}")
    logging.info(f"Train Soundscapes files count: {count_files_in_dir(TRAIN_SOUNDSCAPES_DIR)}")
    logging.info(f"Test Soundscapes files count: {count_files_in_dir(TEST_SOUNDSCAPES_DIR)}")
    # Consider adding checks: does each filename in train_df exist?

    # --- Geographical Analysis ---
    logging.info("--- Geographical Analysis ---")
    location_data = train_df.dropna(subset=['latitude', 'longitude']).copy() # Use a copy for calculations
    location_data = location_data[ # Basic sanity check on coordinates
        (location_data['latitude'].between(-90, 90)) &
        (location_data['longitude'].between(-180, 180))
    ]

    if not location_data.empty:
        # Scatter plot
        plt.figure(figsize=(10, 8))
        sns.scatterplot(x='longitude', y='latitude', data=location_data, alpha=0.3, hue='rating', size='rating', palette='viridis', legend='brief')
        plt.title("Scatter Plot of Recording Locations (Colored by Rating)")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "location_scatter_plot.png")
        plt.show()

        # Geographical outlier detection (using IQR)
        lat_Q1 = location_data['latitude'].quantile(0.25)
        lat_Q3 = location_data['latitude'].quantile(0.75)
        lat_IQR = lat_Q3 - lat_Q1
        lat_lower_bound = lat_Q1 - 1.5 * lat_IQR
        lat_upper_bound = lat_Q3 + 1.5 * lat_IQR

        lon_Q1 = location_data['longitude'].quantile(0.25)
        lon_Q3 = location_data['longitude'].quantile(0.75)
        lon_IQR = lon_Q3 - lon_Q1
        lon_lower_bound = lon_Q1 - 1.5 * lon_IQR
        lon_upper_bound = lon_Q3 + 1.5 * lon_IQR

        logging.info(f"Geographical outlier bounds (IQR): Latitude [{lat_lower_bound:.2f}, {lat_upper_bound:.2f}], Longitude [{lon_lower_bound:.2f}, {lon_upper_bound:.2f}]")

        geo_outliers = location_data[
            (location_data['latitude'] < lat_lower_bound) | (location_data['latitude'] > lat_upper_bound) |
            (location_data['longitude'] < lon_lower_bound) | (location_data['longitude'] > lon_upper_bound)
        ]
        logging.info(f"Number of potential geographical outliers: {geo_outliers.shape[0]}")
        if not geo_outliers.empty:
            logging.info(f"Sample geographical outliers:\n{geo_outliers[['primary_label', 'latitude', 'longitude', 'rating']].head()}")

        # Interactive Map
        plot_geo_locations(train_df) # Pass the original df with rating info

    else:
        logging.warning("Skipping geographical analysis due to missing or invalid location data.")


    # --- Analyze Collection Source ---
    logging.info("--- Analyzing Collection Source ---")
    if 'collection' in train_df.columns:
        plot_distribution(train_df['collection'],
                          title="Distribution of Recordings by Collection Source",
                          xlabel="Collection",
                          ylabel="Count",
                          kind='bar')
        logging.info(f"Collection counts:\n{train_df['collection'].value_counts()}")
    else:
        logging.info("Column 'collection' not found in train_df.")


    # --- Correlation Analysis ---
    logging.info("--- Correlation Analysis ---")
    # Only 'rating' is numerical in the original set, add others if feature engineered (e.g., duration)
    numerical_cols = ['rating', 'num_secondary_labels'] # Add latitude/longitude if desired
    plot_correlation_matrix(train_df, numerical_cols, title="Correlation Matrix of Numerical Features")


    # --- Prepare Submission File (Example) ---
    logging.info("--- Preparing Sample Submission ---")
    # This part just saves the sample submission, actual prediction logic is needed for a real submission
    try:
        submission_path = OUTPUT_DIR / 'submission.csv'
        sample_submission.to_csv(submission_path, index=False)
        logging.info(f"Sample submission file saved to: {submission_path}")
    except Exception as e:
        logging.error(f"Failed to save submission file: {e}")

    logging.info("EDA Script Finished.")


import geopandas as gpd

# --- Geographical Analysis ---
logging.info("--- Geographical Analysis ---")

# Filter valid location data
location_data = train_df.dropna(subset=['latitude', 'longitude']).copy()
location_data = location_data[
    (location_data['latitude'].between(-90, 90)) &
    (location_data['longitude'].between(-180, 180))
]

if not location_data.empty:
    # Load world map
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

    # Scatter plot with world map
    fig, ax = plt.subplots(figsize=(12, 8))
    world.plot(ax=ax, color='lightgrey')  # Background world map
    sns.scatterplot(x='longitude', y='latitude', data=location_data, alpha=0.3,
                    hue='rating', size='rating', palette='viridis', legend='brief', ax=ax)
    plt.title("Scatter Plot of Recording Locations (Colored by Rating)")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "location_scatter_plot.png")
    plt.show()

    # Geographical outlier detection (using IQR)
    lat_Q1, lat_Q3 = location_data['latitude'].quantile([0.25, 0.75])
    lon_Q1, lon_Q3 = location_data['longitude'].quantile([0.25, 0.75])
    lat_IQR, lon_IQR = lat_Q3 - lat_Q1, lon_Q3 - lon_Q1

    lat_lower, lat_upper = lat_Q1 - 1.5 * lat_IQR, lat_Q3 + 1.5 * lat_IQR
    lon_lower, lon_upper = lon_Q1 - 1.5 * lon_IQR, lon_Q3 + 1.5 * lon_IQR

    logging.info(f"Geographical outlier bounds (IQR): Latitude [{lat_lower:.2f}, {lat_upper:.2f}], "
                 f"Longitude [{lon_lower:.2f}, {lon_upper:.2f}]")

    geo_outliers = location_data[
        (location_data['latitude'] < lat_lower) | (location_data['latitude'] > lat_upper) |
        (location_data['longitude'] < lon_lower) | (location_data['longitude'] > lon_upper)
    ]

    logging.info(f"Number of potential geographical outliers: {geo_outliers.shape[0]}")
    if not geo_outliers.empty:
        logging.info(f"Sample geographical outliers:\n{geo_outliers[['primary_label', 'latitude', 'longitude', 'rating']].head()}")

    # Interactive Map
    world_map = folium.Map(location=[location_data['latitude'].mean(), location_data['longitude'].mean()], zoom_start=2)
    marker_cluster = MarkerCluster().add_to(world_map)

    for _, row in location_data.iterrows():
        folium.Marker(
            location=[row['latitude'], row['longitude']],
            popup=f"{row['primary_label']} (Rating: {row['rating']})",
            icon=folium.Icon(color='blue', icon='info-sign')
        ).add_to(marker_cluster)

    map_path = OUTPUT_DIR / "geographical_map.html"
    world_map.save(map_path)
    logging.info(f"Interactive map saved at {map_path}")

else:
    logging.warning("Skipping geographical analysis due to missing or invalid location data.")


sample_submission

