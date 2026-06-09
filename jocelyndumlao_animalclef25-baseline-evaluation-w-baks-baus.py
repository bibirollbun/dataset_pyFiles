import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2  # Import OpenCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tqdm.notebook import tqdm  # For progress bars
import random

import warnings
warnings.filterwarnings("ignore")


# --- Define Evaluation Metrics ---
def calculate_baks_baus(ground_truth, predictions):
    """Calculates BAKS and BAUS scores for animal re-identification.

    Args:
        ground_truth (pd.DataFrame): DataFrame with 'image_id' and 'identity' columns.
        predictions (pd.DataFrame): DataFrame with 'image_id' and 'predicted_identity' columns.

    Returns:
        tuple: (BAKS, BAUS) scores.
    """

    # Merge ground truth and predictions on 'image_id'
    merged_df = pd.merge(ground_truth, predictions, on='image_id', how='inner')
    merged_df.rename(columns={'identity': 'true_identity'}, inplace=True)

    # Identify known and unknown samples
    known_samples = merged_df[merged_df['true_identity'] != 'new_individual']
    unknown_samples = merged_df[merged_df['true_identity'] == 'new_individual']

    # Calculate BAKS (Balanced Accuracy on Known Samples)
    if not known_samples.empty:
        # Calculate accuracy for each individual
        individual_accuracies = []
        for individual in known_samples['true_identity'].unique():
            individual_data = known_samples[known_samples['true_identity'] == individual]
            accuracy = accuracy_score(individual_data['true_identity'], individual_data['predicted_identity'])
            individual_accuracies.append(accuracy)

        BAKS = np.mean(individual_accuracies)
    else:
        BAKS = 0.0

    # Calculate BAUS (Balanced Accuracy on Unknown Samples)
    if not unknown_samples.empty:
        BAUS = accuracy_score(unknown_samples['true_identity'], unknown_samples['predicted_identity'])
    else:
        BAUS = 0.0

    return BAKS, BAUS

def geometric_mean(a, b):
    """Calculates the geometric mean of two numbers."""
    return np.sqrt(a * b)



# --- Define Submission Creation Function ---
def create_submission(image_ids, predictions):
    """Creates a submission DataFrame in the correct format.

    Args:
        image_ids (list): List of image IDs.
        predictions (list): List of predictions corresponding to the image IDs.

    Returns:
        pd.DataFrame: Submission DataFrame.
    """
    submission = pd.DataFrame({'image_id': image_ids, 'identity': predictions})
    return submission



# --- Data Loading and Exploration ---
data_path = '/kaggle/input/animal-clef-2025'
images_path = os.path.join(data_path, 'images')
metadata_path = os.path.join(data_path, 'metadata.csv')
sample_submission_path = os.path.join(data_path, 'sample_submission.csv')

metadata_df = pd.read_csv(metadata_path)
sample_submission_df = pd.read_csv(sample_submission_path)

print("Metadata Head:")
metadata_df.head().style.background_gradient(cmap='plasma')


print("\nSample Submission Head:")
sample_submission_df.head().style.background_gradient(cmap='plasma')


print("\nMetadata Info:")
metadata_df.info()

print("\nValue Counts for 'species' column:")
metadata_df['species'].value_counts()

print("\nValue Counts for 'split' column:")
metadata_df['split'].value_counts()


# --- Visualizations ---

# 1. Distribution of Species
plt.figure(figsize=(10, 6))
ax = sns.countplot(data=metadata_df, x='species', palette="viridis")  # Use a nice palette
plt.title('Distribution of Species', fontsize=16)  # Increase title size
plt.xlabel('Species', fontsize=12)  # Label x-axis
plt.ylabel('Count', fontsize=12)  # Label y-axis
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for readability

# Add value labels on top of the bars
for p in ax.patches:
    ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                textcoords='offset points')

plt.tight_layout()  # Adjust layout to prevent labels from overlapping
plt.show()

# 2. Distribution of Data Split (Query vs. Database)
plt.figure(figsize=(8, 5))
ax = sns.countplot(data=metadata_df, x='split', palette="mako")  # Another nice palette
plt.title('Distribution of Data Split', fontsize=16)
plt.xlabel('Data Split', fontsize=12)
plt.ylabel('Count', fontsize=12)

# Add value labels
for p in ax.patches:
    ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                textcoords='offset points')
plt.tight_layout()
plt.show()

# 3. Image Orientation Distribution
plt.figure(figsize=(8, 6))
ax = sns.countplot(data=metadata_df, x='orientation', palette="crest")  # Yet another palette
plt.title('Image Orientation Distribution', fontsize=16)
plt.xlabel('Orientation', fontsize=12)
plt.ylabel('Count', fontsize=12)

# Add value labels
for p in ax.patches:
    ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                textcoords='offset points')
plt.tight_layout()
plt.show()

# 4. Time Series Analysis (Image Dates)
# Convert 'date' column to datetime objects
metadata_df['date'] = pd.to_datetime(metadata_df['date'])

# Extract year and month
metadata_df['year'] = metadata_df['date'].dt.year
metadata_df['month'] = metadata_df['date'].dt.month

# Plot time series
plt.figure(figsize=(12, 6))
sns.lineplot(data=metadata_df.groupby('year').size(), marker='o', color="#607c8e")  # Specific color
plt.title('Number of Images Over Years', fontsize=16)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Number of Images', fontsize=12)
plt.grid(False)  # Add grid lines for readability
plt.tight_layout()
plt.show()

# 5. Species vs. Split
plt.figure(figsize=(12, 7))
ax = sns.countplot(data=metadata_df, x='species', hue='split', palette="Set2")  # Palette with distinct colors
plt.title('Species Distribution by Split', fontsize=16)
plt.xlabel('Species', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.legend(title='Data Split', fontsize=10)  # Add legend with title

# Add value labels (slightly more complex due to hue)
for p in ax.patches:
    ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', fontsize=8, color='black', xytext=(0, 5),
                textcoords='offset points')
plt.tight_layout()
plt.show()

# 6. Orientation vs. Split
plt.figure(figsize=(10, 6))
ax = sns.countplot(data=metadata_df, x='orientation', hue='split', palette="Paired")  # Another good palette
plt.title('Orientation Distribution by Split', fontsize=16)
plt.xlabel('Orientation', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.legend(title='Data Split', fontsize=10)

# Add value labels
for p in ax.patches:
    ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', fontsize=8, color='black', xytext=(0, 5),
                textcoords='offset points')
plt.tight_layout()
plt.show()



# --- Image Visualization ---
def display_images(metadata, num_images=5):
    """Displays a few sample images from each species."""
    species = metadata['species'].unique()
    num_species = len(species)

    # Adjust subplot grid based on number of species
    fig, axes = plt.subplots(num_species, num_images, figsize=(15, 5 * num_species))

    # If there's only one species, axes will be a 1D array
    if num_species == 1:
        axes = axes[np.newaxis, :]  # Make it a 2D array

    fig.tight_layout()

    for i, sp in enumerate(species):
        species_data = metadata[metadata['species'] == sp]
        if len(species_data) == 0:
            print(f"No images found for species: {sp}")
            continue

        # Check if there are enough images for the sample size
        if len(species_data) < num_images:
            num_to_sample = len(species_data)
        else:
            num_to_sample = num_images

        try:
            species_sample = species_data.sample(num_to_sample)
        except ValueError as e:
            print(f"Error sampling images for species {sp}: {e}")
            continue # Skip to the next species

        for j in range(num_to_sample):
            image_id = species_sample.iloc[j]['image_id']
            image_path = os.path.join(data_path, species_sample.iloc[j]['path'])  # Use corrected path from metadata

            try:
                img = Image.open(image_path)
                ax = axes[i, j]
                ax.imshow(img)
                ax.set_title(f'{sp}')
                ax.axis('off')
            except FileNotFoundError:
                print(f"Image not found: {image_path}")
                ax = axes[i, j]
                ax.text(0.5, 0.5, 'Image Not Found', ha='center', va='center')
                ax.axis('off')
            except Exception as e:
                print(f"Error processing image {image_path}: {e}")
                ax = axes[i, j]
                ax.text(0.5, 0.5, 'Error Loading Image', ha='center', va='center')
                ax.axis('off')
    plt.show()

display_images(metadata_df, num_images=7) # Increased to 7


# --- Simple Baseline Prediction ---

# This is a very basic baseline.  A real solution would involve image analysis.
# This example just predicts "new_individual" for everything as a starting point.

image_ids = sample_submission_df['image_id'].tolist()
predictions = ['new_individual'] * len(image_ids)

baseline_submission = create_submission(image_ids, predictions)

print("\nBaseline Submission Head:")
baseline_submission.head()



# --- Evaluation ---
# Load ground truth identities (for query images only)
ground_truth_df = metadata_df[metadata_df['split'] == 'query'][['image_id', 'identity']]
ground_truth_df.rename(columns={'identity': 'true_identity'}, inplace=True)

# Create a dataframe from baseline submission for evaluation
predictions_df = baseline_submission.copy()
predictions_df.rename(columns={'identity': 'predicted_identity'}, inplace=True)  # Align with evaluate function



# Calculate BAKS and BAUS
baks, baus = calculate_baks_baus(ground_truth_df, predictions_df)
geometric_mean_score = geometric_mean(baks, baus)

print(f"\nBAKS: {baks:.4f}")
print(f"BAUS: {baus:.4f}")
print(f"Geometric Mean: {geometric_mean_score:.4f}")


# --- Submission ---
submission_filename = 'submission.csv'
baseline_submission.to_csv(submission_filename, index=False)
print(f"\nSubmission file saved to: {submission_filename}")
print("\nSubmission file Head:")
baseline_submission.head()




