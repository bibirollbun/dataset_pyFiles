import h5py
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import os
import pickle
from matplotlib.patches import Rectangle

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import BaseEstimator, TransformerMixin

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Model

# New import for image registration:
from skimage.registration import phase_cross_correlation

# -----------------------------
# 1. Functions for Patch Extraction and CNN Features
# -----------------------------
def extract_patch(image, center, patch_size):
    """
    Extract a square patch from the image centered at the given coordinate.
    Assumes image shape is (height, width, channels) and center is (x, y).
    """
    x, y = int(center[0]), int(center[1])
    half_size = patch_size // 2
    # Ensure indices are within bounds
    y_min = max(y - half_size, 0)
    y_max = min(y + half_size, image.shape[0])
    x_min = max(x - half_size, 0)
    x_max = min(x + half_size, image.shape[1])
    patch = image[y_min:y_max, x_min:x_max, :]
    return patch

def extract_cnn_features(patch, cnn_model):
    """
    Resize, preprocess, and extract CNN features from a given image patch.
    
    Parameters:
      patch (ndarray): The image patch to process.
      cnn_model (Model): The pre-trained CNN model for feature extraction.
      
    Returns:
      features (ndarray): Flattened feature vector from the CNN.
    """
    # Resize patch to the input size expected by ResNet50 (e.g., 224x224)
    patch_resized = cv2.resize(patch, (224, 224))
    patch_preprocessed = preprocess_input(np.expand_dims(patch_resized, axis=0))
    features = cnn_model.predict(patch_preprocessed, verbose=0)
    return features.flatten()

# -----------------------------
# 2. Custom Transformer to Extract CNN Features from Patches
# -----------------------------
class PatchFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, image, patch_size, cnn_model):
        """
        Parameters:
          image (ndarray): The whole-slide HE image as a numpy array.
          patch_size (int): Size (in pixels) of the square patch to extract.
          cnn_model (Model): Pre-trained CNN model for feature extraction.
        """
        self.image = image
        self.patch_size = patch_size
        self.cnn_model = cnn_model

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Collect all resized patches in a list
        patches = []
        for coord in X:
            patch = extract_patch(self.image, coord, self.patch_size)
            # Resize patch to the input size expected by ResNet50 (224x224)
            patch_resized = cv2.resize(patch, (224, 224))
            patches.append(patch_resized)
        # Convert list to numpy array: shape (n_samples, 224, 224, channels)
        patches = np.array(patches)
        # Preprocess all patches at once using the appropriate preprocessing function
        patches_preprocessed = preprocess_input(patches.astype(np.float32))
        # Run the CNN model once on the entire batch
        features = self.cnn_model.predict(patches_preprocessed, verbose=0)
        # Flatten features if necessary (ResNet50 with pooling='avg' already outputs 2D arrays)
        features = features.reshape(features.shape[0], -1)
        return features
        
# -----------------------------
# 3. Pipeline Class for the Elucidata Challenge with Caching and Visualization Options
# -----------------------------
class CellTypePipeline:
    """
    Pipeline for loading data, extracting image patch features using a CNN,
    training a multi-output regression model, and generating a submission file.
    
    Optionally, CNN features can be cached (saved/loaded) using pickle to speed up re-runs.
    Additional visualization methods are provided to verify that the spot coordinates 
    and extracted patches align with the HE slide image.
    """
    
    def __init__(self, h5_file_path, patch_size=64):
        self.h5_file_path = h5_file_path
        self.patch_size = patch_size
        self.train_spot_tables = {}
        self.train_images = {}
        self.cell_type_columns = None
        self.cnn_model = None  # To be initialized
        self.feature_extractor_pipeline = None

    def initialize_cnn_model(self):
        """
        Initialize a pre-trained ResNet50 model (without top layers) for feature extraction.
        """
        base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
        self.cnn_model = Model(inputs=base_model.input, outputs=base_model.output)
        print("CNN feature extractor initialized.")

    def load_train_data(self):
        """
        Load training spot data from the H5 file and store each slide as a DataFrame.
        """
        with h5py.File(self.h5_file_path, "r") as f:
            train_spots = f["spots/Train"]
            for slide_name in train_spots.keys():
                spot_array = np.array(train_spots[slide_name])
                df = pd.DataFrame(spot_array, columns=["x", "y"] + [f"C{i}" for i in range(1, 36)])
                self.train_spot_tables[slide_name] = df
        print("Training spot data loaded successfully.")
        
    def load_train_images(self):
        """
        Load training HE images from the H5 file.
        Adjust the key if your H5 file uses a different naming convention.
        """
        with h5py.File(self.h5_file_path, "r") as f:
            # Adjust key if necessary (e.g., f["images/Train"] if needed)
            train_imgs = f["images/Train"]
            for slide_name in train_imgs.keys():
                image_array = np.array(train_imgs[slide_name])
                self.train_images[slide_name] = image_array
        print("Training images loaded successfully.")

    def load_test_data(self, slide_id):
        """
        Load test spot data for a given slide.
        """
        with h5py.File(self.h5_file_path, "r") as f:
            test_spots = f["spots/Test"]
            if slide_id not in test_spots:
                raise ValueError(f"Slide {slide_id} not found in test spot data.")
            spot_array = np.array(test_spots[slide_id])
            test_df = pd.DataFrame(spot_array, columns=["x", "y"])
        print(f"Test spot data for slide {slide_id} loaded successfully.")
        return test_df

    def load_test_image(self, slide_id):
        """
        Load test HE image for a given slide.
        """
        with h5py.File(self.h5_file_path, "r") as f:
            test_imgs = f["images/Test"]
            if slide_id not in test_imgs:
                raise ValueError(f"Slide {slide_id} not found in test images.")
            image_array = np.array(test_imgs[slide_id])
        print(f"Test image for slide {slide_id} loaded successfully.")
        return image_array

    def prepare_training_set(self, slide_id='S_1', cache_path=None):
        """
        Prepare training features and targets for a given slide.
        Uses the HE image to extract patches and then CNN features.
        
        If cache_path is provided and exists, the method will load cached features.
        Otherwise, it will compute the features and then save them to the provided cache_path.
        """
        if cache_path is not None and os.path.exists(cache_path):
            print(f"Loading cached training features from {cache_path} for slide {slide_id} ...")
            with open(cache_path, "rb") as f:
                X_features, y = pickle.load(f)
            return X_features, y
        
        if slide_id not in self.train_spot_tables:
            raise ValueError(f"Slide {slide_id} not found in training spot data.")
        if slide_id not in self.train_images:
            raise ValueError(f"Slide {slide_id} image not loaded.")
            
        df = self.train_spot_tables[slide_id]
        # Assume first two columns are coordinates and the rest are cell type abundances.
        feature_cols = ['x', 'y']
        target_cols = [col for col in df.columns if col not in feature_cols]
        self.cell_type_columns = target_cols  # Assumed consistent across slides
        
        # Extract coordinates (for patch extraction)
        X_coords = df[feature_cols].values.astype(float)
        # Cell type abundance targets
        y = df[target_cols].values.astype(float)
        
        # Build a feature extractor pipeline for this slide.
        he_image = self.train_images[slide_id]
        patch_extractor = PatchFeatureExtractor(he_image, self.patch_size, self.cnn_model)
        self.feature_extractor_pipeline = Pipeline([
            ('patch_extractor', patch_extractor),
            ('scaler', StandardScaler())
        ])
        # Extract features for training
        print(f"Extracting CNN features for slide {slide_id} ...")
        X_features = self.feature_extractor_pipeline.fit_transform(X_coords)
        
        if cache_path is not None:
            print(f"Saving training features for slide {slide_id} to {cache_path} ...")
            with open(cache_path, "wb") as f:
                pickle.dump((X_features, y), f)
                
        print(f"Extracted CNN features for slide {slide_id}.")
        return X_features, y

    def prepare_all_training_set(self, cache_dir=None, align_spots=True):
        """
        Prepare training features and targets for all slides in the training set (S_1 to S_6).
        If cache_dir is provided, each slide will use a separate cache file named train_features_<slide_id>.pkl.
        If align_spots is True, the function computes the optimal alignment for each slide and adjusts
        the spot coordinates accordingly.
        
        Returns:
            X_all (ndarray): Concatenated features for all slides.
            y_all (ndarray): Concatenated target values for all slides.
        """
        X_list = []
        y_list = []
        # Process slides in sorted order (e.g., S_1, S_2, ... S_6)
        for slide_id in sorted(self.train_spot_tables.keys()):
            if align_spots:
                # Compute the optimal alignment shift for the current slide
                #optimal_shift, error, diffphase = self.compute_optimal_shift(slide_id, display=False)
                #print(f"For slide {slide_id}, applying optimal shift {optimal_shift} (error: {error:.4f}).")
                # Adjust the spot coordinates by subtracting the computed shift.
                # (This assumes that the computed shift indicates how much the spots need to be shifted
                # to align with the tissue mask.)
                df = self.train_spot_tables[slide_id]
                coords = df[['x', 'y']].values.astype(float)
                #adjusted_coords = coords - optimal_shift  # shift the spots into alignment
                #df[['x', 'y']] = adjusted_coords
                self.train_spot_tables[slide_id] = df  # update the stored DataFrame
    
            slide_cache_path = os.path.join(cache_dir, f"train_features_{slide_id}.pkl") if cache_dir else None
            X, y = self.prepare_training_set(slide_id=slide_id, cache_path=slide_cache_path)
            X_list.append(X)
            y_list.append(y)
        X_all = np.concatenate(X_list, axis=0)
        y_all = np.concatenate(y_list, axis=0)
        print("All training features extracted and concatenated.")
        return X_all, y_all

    def build_regression_pipeline(self):
        """
        Build and return a regression pipeline that uses the pre-extracted CNN features.
        """
        pipeline = Pipeline([
            ('regressor', MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42)))
        ])
        return pipeline

    def train(self, X, y):
        """
        Train the regression model on the provided features and targets.
        """
        reg_pipeline = self.build_regression_pipeline()
        reg_pipeline.fit(X, y)
        print("Regression model training complete.")
        return reg_pipeline

    def predict(self, reg_model, X_test):
        """
        Predict cell type abundances on test features.
        """
        predictions = reg_model.predict(X_test)
        return predictions

    def create_submission(self, test_df, predictions, submission_filename="submission.csv"):
        """
        Create a submission CSV file with predicted cell type abundances.
        """
        pred_df = pd.DataFrame(predictions, columns=self.cell_type_columns, index=test_df.index)
        pred_df.insert(0, 'ID', pred_df.index)
        pred_df.to_csv(submission_filename, index=False)
        print(f"Submission file '{submission_filename}' created!")

    # -----------------------------
    # Visualization Methods
    # -----------------------------
    def visualize_spot_overlay(self, slide_id, flip_y=False):
        """
        Visualize the overlay of spot coordinates on the HE slide image.
        Optionally, flip the y-axis if needed.
        """
        if slide_id not in self.train_images or slide_id not in self.train_spot_tables:
            raise ValueError(f"Slide {slide_id} data not found.")
        image = self.train_images[slide_id]
        df = self.train_spot_tables[slide_id]
        coords = df[['x', 'y']].values.astype(float)
        if flip_y:
            coords[:, 1] = image.shape[0] - coords[:, 1]
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        plt.scatter(coords[:, 0], coords[:, 1], marker='o', color='red', s=25)
        plt.title(f"Overlay of Spot Coordinates for Slide {slide_id}")
        plt.show()

    def visualize_extracted_patches(self, slide_id, num_patches=5, flip_y=False):
        """
        Visualize a few extracted patches from the slide to verify correct extraction.
        """
        if slide_id not in self.train_images or slide_id not in self.train_spot_tables:
            raise ValueError(f"Slide {slide_id} data not found.")
        image = self.train_images[slide_id]
        df = self.train_spot_tables[slide_id]
        coords = df[['x', 'y']].values.astype(float)
        if flip_y:
            coords[:, 1] = image.shape[0] - coords[:, 1]
        fig, axes = plt.subplots(1, num_patches, figsize=(num_patches * 3, 3))
        for i in range(num_patches):
            patch = extract_patch(image, coords[i], self.patch_size)
            axes[i].imshow(patch)
            axes[i].set_title(f"Patch {i}")
            axes[i].axis("off")
        plt.suptitle(f"Extracted Patches for Slide {slide_id}")
        plt.show()
    
    def visualize_cnn_input(self, slide_id, index=0, flip_y=False):
        """
        Visualize the resized patch for a given spot index with an overlay of the spot position,
        plot the distribution of the 35 cell type abundancies for that spot,
        print the patch's top-left (X, Y) coordinates, and display the full slide image
        with a rectangle marking the patch position.
        """
        if slide_id not in self.train_images or slide_id not in self.train_spot_tables:
            raise ValueError(f"Slide {slide_id} data not found.")
            
        # Get image and corresponding DataFrame
        image = self.train_images[slide_id]
        df = self.train_spot_tables[slide_id]
        
        # Get coordinates from the DataFrame
        coords = df[['x', 'y']].values.astype(float)
        if flip_y:
            coords[:, 1] = image.shape[0] - coords[:, 1]
        coord = coords[index]
        
        # Compute patch boundaries and extract patch
        half_size = self.patch_size // 2
        x = int(coord[0])
        y = int(coord[1])
        x_min = max(x - half_size, 0)
        y_min = max(y - half_size, 0)
        patch = extract_patch(image, coord, self.patch_size)
        
        # Resize patch to CNN input size (e.g., 224x224)
        patch_resized = cv2.resize(patch, (224, 224))
        
        # Compute the spot's relative position within the patch
        rel_x = x - x_min
        rel_y = y - y_min
        scale_x = 224 / patch.shape[1]
        scale_y = 224 / patch.shape[0]
        spot_resized_x = rel_x * scale_x
        spot_resized_y = rel_y * scale_y
    
        # Retrieve cell type abundancies for this spot.
        abundances = df.iloc[index][[col for col in df.columns if col not in ['x', 'y']]]
        
        # Print the patch's top-left coordinates
        print(f"Patch top-left coordinates: (x_min: {x_min}, y_min: {y_min})")
        
        # Create three subplots: left for the full patch, middle for resized patch, right for the bar chart.
        fig, (ax_full, ax_img, ax_bar) = plt.subplots(1, 3, figsize=(18, 5))

        # Right subplot: Full slide image with a rectangle overlay marking the patch location.
        ax_full.imshow(image)
        # Draw a rectangle at (x_min, y_min) with width and height equal to patch_size.
        rect = Rectangle((x_min, y_min), self.patch_size, self.patch_size, linewidth=2, edgecolor='red', facecolor='none')
        ax_full.add_patch(rect)
        ax_full.set_title("Full Slide with Patch Overlay")
        ax_full.axis("off")
        
        # Middle subplot: Resized patch with spot overlay.
        ax_img.imshow(patch_resized)
        ax_img.scatter([spot_resized_x], [spot_resized_y], marker='x', color='red', s=50)
        ax_img.set_title(f"Resized Patch (Index {index})")
        # Annotate the patch with its top-left coordinates.
        ax_img.text(5, 20, f"({x_min}, {y_min})", color='yellow', fontsize=12, 
                    bbox=dict(facecolor='black', alpha=0.5))
        ax_img.axis("off")
        
        # Left subplot: Bar chart of the cell type abundancies.
        cell_types = abundances.index.tolist()  # e.g., ['C1', 'C2', ..., 'C35']
        ax_bar.bar(cell_types, abundances.values)
        ax_bar.set_title("Cell Type Abundance Distribution")
        ax_bar.set_xticklabels(cell_types, rotation=90)
        ax_bar.set_ylabel("Abundance")
        
        plt.tight_layout()
        plt.show()
            
    def compute_optimal_shift(self, slide_id, flip_y=False, upsample_factor=10, spot_radius=3, display=False):
        """
        Compute the optimal translational shift to align the spot coordinates with the tissue image.
        
        This function creates a binary tissue mask from the HE image (using Otsu thresholding)
        and a corresponding binary spot mask by drawing filled circles at each spot coordinate.
        It then computes the optimal (sub-pixel) shift between these masks using phase cross-correlation.
        
        Parameters:
          slide_id (str): Identifier of the slide to process.
          flip_y (bool): If True, flip the y-axis of spot coordinates.
          upsample_factor (int): Upsampling factor for sub-pixel precision.
          spot_radius (int): Radius (in pixels) for drawing spots in the mask.
          display (bool): If True, display the tissue mask, spot mask, their overlay, and the HE image with corrected spots.
          
        Returns:
          optimal_shift (ndarray): Optimal shift as (x_shift, y_shift).
          error (float): Registration error.
          diffphase (float): Diffphase value from phase_cross_correlation.
        """
        if slide_id not in self.train_images or slide_id not in self.train_spot_tables:
            raise ValueError(f"Slide {slide_id} data not found.")
        
        # Retrieve image and spot coordinates
        image = self.train_images[slide_id]
        df = self.train_spot_tables[slide_id]
        coords = df[['x', 'y']].values.astype(float)
        if flip_y:
            coords[:, 1] = image.shape[0] - coords[:, 1]
        
        # Convert image to grayscale if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            image_gray = image.copy()
        
        # Convert grayscale image to uint8 if needed (required for Otsu thresholding)
        if image_gray.dtype != np.uint8:
            image_gray = cv2.normalize(image_gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                
        # Generate tissue mask using Otsu thresholding
        _, tissue_mask = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Create a blank mask for the spots (same dimensions as tissue_mask)
        spot_mask = np.zeros_like(tissue_mask, dtype=np.uint8)
        for pt in coords:
            x, y = int(round(pt[0])), int(round(pt[1]))
            cv2.circle(spot_mask, (x, y), radius=spot_radius, color=255, thickness=-1)
        
        # Compute the shift using phase cross-correlation
        from skimage.registration import phase_cross_correlation
        shift, error, diffphase = phase_cross_correlation(tissue_mask, spot_mask, upsample_factor=upsample_factor)
        # shift is returned as (row_shift, col_shift) => (y, x)
        optimal_shift = np.array([shift[1], shift[0]])  # Convert to (x_shift, y_shift)
        print("Optimal shift (x, y):", optimal_shift)
        print("Registration error:", error)
        
        if display:
            # Display tissue mask, spot mask, and overlay
            plt.figure(figsize=(15, 5))
            plt.subplot(1, 3, 1)
            plt.title("Tissue Mask")
            plt.imshow(tissue_mask, cmap='gray')
            plt.subplot(1, 3, 2)
            plt.title("Spot Mask")
            plt.imshow(spot_mask, cmap='gray')
            plt.subplot(1, 3, 3)
            plt.title("Overlay of Tissue and Spots")
            plt.imshow(tissue_mask, cmap='gray')
            plt.imshow(spot_mask, cmap='jet', alpha=0.5)
            plt.tight_layout()
            plt.show()
            
            # Plot the original HE image with corrected spot coordinates
            adjusted_coords = coords - optimal_shift
            plt.figure(figsize=(10, 10))
            plt.imshow(image)
            plt.scatter(adjusted_coords[:, 0], adjusted_coords[:, 1], marker='o', color='lime', s=25)
            plt.title("Corrected Spots Overlay on Tissue Image")
            plt.show()
        
        return optimal_shift, error, diffphase

    def manual_shift_alignment(self, slide_id, x_shift, y_shift, flip_y=False, display=True):
        """
        Manually apply a specified x,y shift to the spot coordinates for a given slide,
        and plot the image with original and shifted spot overlays.
        
        Parameters:
          slide_id (str): Identifier of the slide.
          x_shift (float): Shift to apply in the x-direction (positive shifts right).
          y_shift (float): Shift to apply in the y-direction (positive shifts down).
          flip_y (bool): If True, flip the y-axis of the spot coordinates.
          display (bool): If True, display the plots.
          
        Returns:
          original_coords (ndarray): The original spot coordinates.
          shifted_coords (ndarray): The adjusted spot coordinates.
        """
        if slide_id not in self.train_images or slide_id not in self.train_spot_tables:
            raise ValueError(f"Slide {slide_id} data not found.")
        
        # Retrieve the image and spot DataFrame
        image = self.train_images[slide_id]
        df = self.train_spot_tables[slide_id]
        
        # Extract original coordinates
        original_coords = df[['x', 'y']].values.astype(float)
        if flip_y:
            original_coords[:, 1] = image.shape[0] - original_coords[:, 1]
        
        # Compute the shifted coordinates: here we add the shift.
        # (Adjust as needed if your coordinate system requires subtracting the shift.)
        shift_vector = np.array([x_shift, y_shift])
        shifted_coords = original_coords + shift_vector
        
        if display:
            plt.figure(figsize=(14, 7))
            
            # Plot original overlay
            plt.subplot(1, 2, 1)
            plt.imshow(image)
            plt.scatter(original_coords[:, 0], original_coords[:, 1], 
                        marker='o', color='red', s=25, label='Original Spots')
            plt.title(f"Slide {slide_id} - Original Spots")
            plt.legend()
            plt.axis("off")
            
            # Plot shifted overlay
            plt.subplot(1, 2, 2)
            plt.imshow(image)
            plt.scatter(shifted_coords[:, 0], shifted_coords[:, 1], 
                        marker='o', color='lime', s=25, label='Shifted Spots')
            plt.title(f"Slide {slide_id} - Shifted Spots\n(x_shift: {x_shift}, y_shift: {y_shift})")
            plt.legend()
            plt.axis("off")
            
            plt.tight_layout()
            plt.show()
            
        return original_coords, shifted_coords





# -----------------------------
# 4. Example Usage with Caching and Visualization Options
# -----------------------------

# Path to the provided H5 data file
h5_file_path = "/kaggle/input/el-hackathon-2025/elucidata_ai_challenge_data.h5"

# Optionally specify a directory for caching training features (for slides S_1 to S_6)
train_cache_dir = "train_features_cache"
os.makedirs(train_cache_dir, exist_ok=True)
test_cache_path = "test_features_S_7.pkl"      # For slide S_7 test features

# Initialize the pipeline with desired patch size (in pixels)
pipeline_obj = CellTypePipeline(h5_file_path, patch_size=64)

# Initialize the CNN feature extractor (ResNet50)
pipeline_obj.initialize_cnn_model()

# Load training spots and images
pipeline_obj.load_train_data()
pipeline_obj.load_train_images()




# Optional: Visualize the overlay of spot coordinates on a training slide (e.g., S_1)
pipeline_obj.visualize_spot_overlay(slide_id='S_1', flip_y=False)





pipeline_obj.compute_optimal_shift(slide_id="S_1", display=True)


pipeline_obj.manual_shift_alignment(slide_id="S_1", x_shift=-50, y_shift=-50)



# Optional: Visualize a few extracted patches from a training slide (e.g., S_1)
pipeline_obj.visualize_extracted_patches(slide_id='S_1', num_patches=5, flip_y=False)




# Optional: Visualize the CNN input (resized patch) for a specific spot (e.g., index 0 from S_1)
pipeline_obj.visualize_cnn_input(slide_id='S_1', index=0, flip_y=False)


# Optional: Visualize the CNN input (resized patch) for a specific spot (e.g., index 0 from S_1)
pipeline_obj.visualize_cnn_input(slide_id='S_1', index=1, flip_y=False)


pipeline_obj.visualize_cnn_input(slide_id='S_1', index=10, flip_y=False)



pipeline_obj.visualize_spot_overlay(slide_id='S_2', flip_y=False)
pipeline_obj.visualize_extracted_patches(slide_id='S_2', num_patches=5, flip_y=False)



pipeline_obj.compute_optimal_shift(slide_id="S_2", display=True)


pipeline_obj.manual_shift_alignment(slide_id="S_2", x_shift=-60, y_shift=-60)


pipeline_obj.visualize_cnn_input(slide_id='S_2', index=0, flip_y=False)



pipeline_obj.visualize_cnn_input(slide_id='S_3', index=0, flip_y=False)



pipeline_obj.visualize_cnn_input(slide_id='S_4', index=0, flip_y=False)



pipeline_obj.visualize_cnn_input(slide_id='S_5', index=0, flip_y=False)



pipeline_obj.visualize_spot_overlay(slide_id='S_2', flip_y=False)
pipeline_obj.visualize_extracted_patches(slide_id='S_2', num_patches=5, flip_y=False)



pipeline_obj.compute_optimal_shift(slide_id="S_3", display=True)


pipeline_obj.compute_optimal_shift(slide_id="S_4", display=True)


pipeline_obj.compute_optimal_shift(slide_id="S_5", display=True)


pipeline_obj.compute_optimal_shift(slide_id="S_6", display=True)


#pipeline_obj.visualize_cnn_input(slide_id='S_7', index=0, flip_y=False)



skip_training = False

if not skip_training:
    # Prepare training features and targets from all slides (S_1 to S_6)
    X_train, y_train = pipeline_obj.prepare_all_training_set(cache_dir=train_cache_dir)





if not skip_training:
    X_train


import os
import pickle
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier


def convert_to_rank(arr):
    """
    Converts each row in the array to a rank order.
    The highest value gets rank 1, the second highest gets rank 2, and so on.
    
    Parameters:
      arr (numpy.ndarray): 2D array with shape (n_samples, 35) of absolute abundances.
      
    Returns:
      numpy.ndarray: 2D array of the same shape with rank orders.
    """
    return np.argsort(-arr, axis=1) + 1



if not skip_training:

    # Prepare training features and targets (absolute abundances) from slides S_1 to S_6
    X_train, y_train = pipeline_obj.prepare_all_training_set(cache_dir=train_cache_dir)
    
    # Convert the absolute abundance targets to rank order (1 = highest abundance)
    y_train_rank = convert_to_rank(y_train)
    
    # Convert the rank order to quartile bins:
    # Ranks 1-5   -> ceil(rank/5) = 1, so quartile = 8 - 1 = 7 (top abundant)
    # Ranks 6-10  -> ceil(rank/5) = 2, so quartile = 8 - 2 = 6
    # ...
    # Ranks 31-35 -> ceil(rank/5) = 7, so quartile = 8 - 7 = 1 (least abundant)
    y_train_quartile = (8 - np.ceil(y_train_rank / 5)).astype(int)
    
    # Train a model on the extracted CNN features with quartile targets
    reg_model = pipeline_obj.train(X_train, y_train)
    
    # Load test data and image for slide S_7 (as per challenge description)
    test_df = pipeline_obj.load_test_data(slide_id='S_7')
    test_image = pipeline_obj.load_test_image(slide_id='S_7')



if not skip_training:
    # Build a feature extractor for test slide using its HE image
    test_patch_extractor = PatchFeatureExtractor(test_image, pipeline_obj.patch_size, pipeline_obj.cnn_model)
    test_feature_pipeline = Pipeline([
        ('patch_extractor', test_patch_extractor),
        ('scaler', StandardScaler())
    ])
    X_test_coords = test_df[['x', 'y']].values.astype(float)




if not skip_training:
    # Check for cached test features
    if os.path.exists(test_cache_path):
        print(f"Loading cached test features from {test_cache_path} ...")
        with open(test_cache_path, "rb") as f:
            X_test_features = pickle.load(f)
    else:
        print("Extracting CNN features for test data ...")
        X_test_features = test_feature_pipeline.fit_transform(X_test_coords)
        print(f"Saving test features to {test_cache_path} ...")
        with open(test_cache_path, "wb") as f:
            pickle.dump(X_test_features, f)



if not skip_training:

    # Predict cell type abundances for test data
    predictions = pipeline_obj.predict(reg_model, X_test_features)
    
    # Create submission file
    pipeline_obj.create_submission(test_df, predictions, submission_filename="submission_model1.csv")


import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import pandas as pd

# -------------------------------
# Configuration
# -------------------------------
# Assume submission_df has an "ID" column and cell type prediction columns (e.g. "C1", "C2", ..., "C35")
# Here, we extract the CNN predictions as a numpy array.
# Adjust the column selection if needed.
prediction_columns = submission_df.columns.drop("ID")
cnn_predictions = submission_df[prediction_columns].values
num_cell_types = cnn_predictions.shape[1]  # e.g., 35
noise_dim = 10  # Dimension of noise input
input_dim = cnn_predictions.shape[1]

# -------------------------------
# Build Generator Model
# -------------------------------
def build_generator(input_dim, num_cell_types, noise_dim):
    # The generator takes the CNN prediction and a noise vector and outputs a refined prediction.
    cnn_input = layers.Input(shape=(input_dim,), name="cnn_prediction")
    noise_input = layers.Input(shape=(noise_dim,), name="noise")
    x = layers.Concatenate()([cnn_input, noise_input])
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dense(64, activation='relu')(x)
    refined_prediction = layers.Dense(num_cell_types, activation='linear', name="refined_prediction")(x)
    model = models.Model(inputs=[cnn_input, noise_input], outputs=refined_prediction, name="Generator")
    return model

# -------------------------------
# Build Discriminator Model
# -------------------------------
def build_discriminator(num_cell_types):
    # The discriminator distinguishes between real cell type labels and refined predictions.
    inp = layers.Input(shape=(num_cell_types,), name="cell_type_prediction")
    x = layers.Dense(64, activation='relu')(inp)
    x = layers.Dense(64, activation='relu')(x)
    validity = layers.Dense(1, activation='sigmoid', name="validity")(x)
    model = models.Model(inputs=inp, outputs=validity, name="Discriminator")
    return model

# Instantiate the models
generator = build_generator(input_dim, num_cell_types, noise_dim)
discriminator = build_discriminator(num_cell_types)

# Compile the discriminator
discriminator.compile(
    optimizer=optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# -------------------------------
# Build Combined GAN Model
# -------------------------------
# Freeze the discriminator weights when training the generator.
discriminator.trainable = False

cnn_pred_input = layers.Input(shape=(input_dim,), name="cnn_input")
noise_input = layers.Input(shape=(noise_dim,), name="noise_input")
refined_pred = generator([cnn_pred_input, noise_input])
validity = discriminator(refined_pred)

combined_model = models.Model(
    inputs=[cnn_pred_input, noise_input],
    outputs=[refined_pred, validity],
    name="Combined_GAN"
)

# Use a weighted combination of a content loss (MSE) and the adversarial loss.
lambda_content = 100  # Adjust this weight as needed

combined_model.compile(
    optimizer=optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
    loss={'refined_prediction': 'mse', 'validity': 'binary_crossentropy'},
    loss_weights={'refined_prediction': lambda_content, 'validity': 1.0}
)

# -------------------------------
# GAN Training (Demonstration)
# -------------------------------
# In a realistic setup, you would train the GAN using your training set (with true labels).
# For demonstration purposes, we will use cnn_predictions as a proxy for ground truth.
# Replace gt_batch with your actual training labels when available.
epochs = 1000
batch_size = 32
n_samples = cnn_predictions.shape[0]
real_labels = np.ones((batch_size, 1))
fake_labels = np.zeros((batch_size, 1))

for epoch in range(epochs):
    # ---------------------
    # Train Discriminator
    # ---------------------
    idx = np.random.randint(0, n_samples, batch_size)
    cnn_preds_batch = cnn_predictions[idx]
    # In practice, use your true labels for the cell type abundances:
    gt_batch = cnn_preds_batch  # Replace this with your ground truth labels
    
    noise = np.random.normal(0, 1, (batch_size, noise_dim))
    refined_preds = generator.predict([cnn_preds_batch, noise], verbose=0)
    
    d_loss_real = discriminator.train_on_batch(gt_batch, real_labels)
    d_loss_fake = discriminator.train_on_batch(refined_preds, fake_labels)
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
    
    # ---------------------
    # Train Generator (via the combined model)
    # ---------------------
    noise = np.random.normal(0, 1, (batch_size, noise_dim))
    g_loss = combined_model.train_on_batch(
        [cnn_preds_batch, noise],
        {'refined_prediction': gt_batch, 'validity': real_labels}
    )
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch} [D loss: {d_loss[0]:.4f}, acc.: {100*d_loss[1]:.2f}%] [G loss: {g_loss[0]:.4f}]")

# -------------------------------
# Refine the CNN Predictions
# -------------------------------
# Once the GAN is trained (or if you load pre-trained weights), refine the predictions.
noise_new = np.random.normal(0, 1, (n_samples, noise_dim))
refined_predictions = generator.predict([cnn_predictions, noise_new], verbose=0)

# Create a new submission DataFrame with the refined predictions.
# We assume the submission_df has an "ID" column; update the prediction columns.
refined_submission_df = submission_df.copy()
refined_submission_df[prediction_columns] = pd.DataFrame(
    refined_predictions, index=refined_submission_df.index, columns=prediction_columns
)

print("Refined submission_df (head):")
print(refined_submission_df.head())

# Optionally, save the refined submission to CSV.
refined_submission_df.to_csv("submission.csv", index=False)


