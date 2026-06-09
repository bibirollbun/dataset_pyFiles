import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm  # For progress bars
from skimage import feature  # For Canny edge detection
import logging
import traceback
from collections import defaultdict



# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# Data Loading and Inspection 
try:
    train_labels = pd.read_csv('/kaggle/input/image-matching-challenge-2025/train_labels.csv')
    train_thresholds = pd.read_csv('/kaggle/input/image-matching-challenge-2025/train_thresholds.csv')
    submission_df = pd.read_csv('/kaggle/input/image-matching-challenge-2025/sample_submission.csv')
    logging.info("Data loaded successfully.")

    # --- Data Inspection Limit ---
    INSPECTION_LIMIT = 5  # Limit the number of rows printed for inspection

    print("Train Labels Head:")
    print(train_labels.head(INSPECTION_LIMIT))
    print("\nTrain Thresholds Head:")
    print(train_thresholds.head(INSPECTION_LIMIT))
    print("\nSubmission DataFrame Head:")
    print(submission_df.head(INSPECTION_LIMIT))

except FileNotFoundError as e:
    logging.error(f"Error loading data: {e}")
    raise  # Re-raise the exception to halt execution
except Exception as e:
    logging.error(f"An unexpected error occurred during data loading: {e}")
    logging.error(traceback.format_exc())  # Log the traceback for debugging
    raise



# Visualization Functions 
def visualize_image(image_path, title="Original"):
    """Displays an image."""
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image at path: {image_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB for matplotlib
        plt.imshow(img)
        plt.title(title)
        plt.axis('off')
        plt.show()
    except Exception as e:
        logging.error(f"Error visualizing image {image_path}: {e}")

def visualize_grayscale(image_path):
    """Displays the grayscale version of an image."""
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not read image at path: {image_path}")
        plt.imshow(img, cmap='gray')
        plt.title("Grayscale")
        plt.axis('off')
        plt.show()
    except Exception as e:
        logging.error(f"Error visualizing grayscale image {image_path}: {e}")

def visualize_canny_edges(image_path, sigma=1):
    """Displays Canny edge detection result."""
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not read image at path: {image_path}")
        edges = feature.canny(img, sigma=sigma)  # Apply Canny edge detection
        plt.imshow(edges, cmap='gray')
        plt.title(f"Canny Edges (Sigma={sigma})")
        plt.axis('off')
        plt.show()
    except Exception as e:
        logging.error(f"Error visualizing Canny edges for image {image_path}: {e}")

def visualize_all(image_path, sigma=1):
    """Visualizes original, grayscale, and Canny edges of an image."""
    try:
        plt.figure(figsize=(15, 5))  # Adjust figure size

        # Original Image
        plt.subplot(1, 3, 1)
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image at path: {image_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img)
        plt.title("Original")
        plt.axis('off')

        # Grayscale Image
        plt.subplot(1, 3, 2)
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        plt.imshow(img, cmap='gray')
        plt.title("Grayscale")
        plt.axis('off')

        # Canny Edges
        plt.subplot(1, 3, 3)
        edges = feature.canny(img, sigma=sigma)
        plt.imshow(edges, cmap='gray')
        plt.title(f"Canny Edges (Sigma={sigma})")
        plt.axis('off')

        plt.tight_layout()  # Adjust subplot parameters for a tight layout.
        plt.show()
    except Exception as e:
        logging.error(f"Error visualizing all images for {image_path}: {e}")




# Example Visualizations
# Choose an example image from the training data.
try:
    example_image_path = '/kaggle/input/image-matching-challenge-2025/train/amy_gardens/peach_0001.png'
    # Visualize the example image
    visualize_all(example_image_path, sigma=1.5)

    example_image_path2 = '/kaggle/input/image-matching-challenge-2025/train/fbk_vineyard/vineyard_split_1_frame_0905.png'
    visualize_all(example_image_path2, sigma=1.5)

    example_image_path3 = '/kaggle/input/image-matching-challenge-2025/train/imc2023_heritage/cyprus_dsc_6496.png'
    visualize_all(example_image_path3, sigma=1.5)
except FileNotFoundError as e:
    logging.warning(f"Example image file not found: {e}. Skipping example visualizations.")
except Exception as e:
    logging.error(f"Error during example visualizations: {e}")



# Keypoint Detection and Matching

def detect_and_match_orb(image_path1, image_path2):
    """Detects ORB keypoints and matches them between two images."""
    try:
        img1 = cv2.imread(image_path1, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(image_path2, cv2.IMREAD_GRAYSCALE)

        if img1 is None or img2 is None:
            raise ValueError(f"Could not read one or both images: {image_path1}, {image_path2}")

        # Initiate ORB detector
        orb = cv2.ORB_create()

        # Find the keypoints and descriptors with ORB
        kp1, des1 = orb.detectAndCompute(img1, None)
        kp2, des2 = orb.detectAndCompute(img2, None)

        if kp1 is None or kp2 is None or len(kp1) == 0 or len(kp2) == 0:
            logging.warning(f"No keypoints found in one or both images: {image_path1}, {image_path2}. Skipping matching.")
            return

        # Create BFMatcher object (Brute-Force Matcher)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Match descriptors
        matches = bf.match(des1, des2)

        # Sort them in the order of their distance
        matches = sorted(matches, key=lambda x: x.distance)

        # Draw first 20 matches.
        img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:20], None,
                               flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

        plt.imshow(img3)
        plt.title("ORB Feature Matching (Top 20)")
        plt.axis('off')
        plt.show()

    except Exception as e:
        logging.error(f"Error detecting and matching ORB features: {e}")


# Example usage of keypoint matching
try:
    example_image_path1 = '/kaggle/input/image-matching-challenge-2025/train/amy_gardens/peach_0001.png'
    example_image_path2 = '/kaggle/input/image-matching-challenge-2025/train/amy_gardens/peach_0002.png'  # Assuming a second image exists
    detect_and_match_orb(example_image_path1, example_image_path2)
except FileNotFoundError as e:
    logging.warning(f"Example image file not found for ORB matching: {e}. Skipping ORB matching.")
except Exception as e:
    logging.error(f"Error during ORB feature matching example: {e}")



# --- 5.  Pose Prediction (Placeholder) ---

def predict_pose(image_path):
    """Placeholder function for pose prediction."""
    # This is where you would integrate we SfM/COLMAP pipeline.
    # For now, let's return a dummy pose.
    rotation_matrix = np.eye(3).flatten()  # Identity matrix (no rotation)
    translation_vector = np.zeros(3)  # No translation
    return rotation_matrix, translation_vector


# Submission File Generation 

def generate_submission(submission_df, output_csv="submission.csv"):
    """Generates a submission file with dummy pose predictions."""
    submission_data = []
    failed_images = []
    dataset_scene_counts = defaultdict(int)  # Track scene counts for each dataset

    for index, row in tqdm(submission_df.iterrows(), total=len(submission_df), desc="Generating Submission"):
        image_id = row['image_id']
        dataset = row['dataset']
        scene = row['scene']
        image_name = row['image']  # Extract the image name from the 'image' column.  

        image_path = os.path.join('/kaggle/input/image-matching-challenge-2025/test', dataset, scene, image_name)

        try:
            # Predict the pose using the placeholder function.
            rotation_matrix, translation_vector = predict_pose(image_path)

            # Format the pose as strings.
            rotation_string = ";".join(map(str, rotation_matrix))
            translation_string = ";".join(map(str, translation_vector))

            # Append the data to the list.
            submission_data.append({
                'image_id': image_id,
                'dataset': dataset,
                'scene': scene,
                'image': image_name,  # Keep the image name as is.
                'rotation_matrix': rotation_string,
                'translation_vector': translation_string
            })
            dataset_scene_counts[dataset] += 1 # increment the image count per dataset
        except FileNotFoundError:
            logging.error(f"Image file not found: {image_path}.  Setting pose to NaN.")
            rotation_string = ";".join(['nan'] * 9)
            translation_string = ";".join(['nan'] * 3)
            submission_data.append({
                'image_id': image_id,
                'dataset': dataset,
                'scene': 'outliers',
                'image': image_name,  # Keep the image name as is.
                'rotation_matrix': rotation_string,
                'translation_vector': translation_string
            })
            failed_images.append(image_id) #keep track of failed images
        except Exception as e:
            logging.error(f"Error processing image {image_path}: {e}")
            logging.error(traceback.format_exc()) # Log the traceback for more details

            #In case of an error, put it into outliers
            rotation_string = ";".join(['nan'] * 9)
            translation_string = ";".join(['nan'] * 3)
            submission_data.append({
                'image_id': image_id,
                'dataset': dataset,
                'scene': 'outliers',
                'image': image_name,  # Keep the image name as is.
                'rotation_matrix': rotation_string,
                'translation_vector': translation_string
            })

            failed_images.append(image_id)

    # Create a new DataFrame from the submission data.
    submission_df = pd.DataFrame(submission_data)

    # Save the DataFrame to a CSV file.
    submission_df.to_csv(output_csv, index=False)
    logging.info(f"Submission file saved to {output_csv}")

    if failed_images:
        logging.warning(f"Failed to process {len(failed_images)} images.  They were assigned to outliers.")
        logging.warning(f"Failed image IDs: {failed_images}")

    # Log number of images per dataset to check the data distribution
    for dataset, count in dataset_scene_counts.items():
        logging.info(f"Number of images processed for dataset {dataset}: {count}")


try:
    #Ensure the submission_df exists
    if 'submission_df' not in locals():
        submission_df = pd.read_csv('/kaggle/input/image-matching-challenge-2025/sample_submission.csv') #or wherever it is
    generate_submission(submission_df)
except Exception as e:
    logging.critical(f"Critical error during submission generation: {e}")
    logging.critical(traceback.format_exc())  # Log the traceback for debugging


submission_df.head()




