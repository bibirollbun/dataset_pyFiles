import os
import gc
import time
import math
import cv2
from pathlib import Path

import numpy as np
import pandas as pd
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from tqdm.auto import tqdm


class CFG:
    # Data Paths
    test_soundscapes: str = "/kaggle/input/birdclef-2025/test_soundscapes"
    submission_csv: str = "/kaggle/input/birdclef-2025/sample_submission.csv"
    taxonomy_csv: str = "/kaggle/input/birdclef-2025/taxonomy.csv"
    model_path: str = "/kaggle/input/efficientnet-b0-pytorch-train"

    # Audio parameters
    # Sampling rate for audio processing (samples per second).
    FS: int = 32000  
    # The duration of each audio segment to be processed in seconds.
    WINDOW_SIZE: int = 5  

    # Mel spectrogram parameters
    # The number of FFT components used to compute the spectrogram.
    N_FFT: int = 1024
    # The number of samples between successive frames in the spectrogram.
    HOP_LENGTH: int = 512
    # The number of Mel bands to generate.
    N_MELS: int = 128
    # The minimum frequency (in Hz) to include in the Mel spectrogram.
    FMIN: int = 50
    # The maximum frequency (in Hz) to include in the Mel spectrogram.
    FMAX: int = 14000
    # The target shape (height, width) for the mel spectrogram image after resizing.
    TARGET_SHAPE: tuple[int, int] = (256, 256)

    # Model parameters
    # The name of the base model architecture to use (e.g., "efficientnet_b0").
    model_name: str = "efficientnet_b0"
    # The number of input channels for the model (1 for grayscale/mel spectrogram).
    in_channels: int = 1
    # The device to use for inference ("cuda" for GPU, "cpu" for CPU).
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Inference parameters
    # The number of audio segments to process in a single batch during inference.
    batch_size: int = 16
    # Flag to enable or disable Test-Time Augmentation (TTA).
    use_tta: bool = False  
    # The number of TTA variations to apply if use_tta is True.
    tta_count: int = 3   
    # The probability threshold to consider a species as detected in a segment.
    threshold: int = 0.5

    # Model selection
    # Flag to use only models from specific folds if True. If False, all found models are used.
    use_specific_folds: bool = False
    # A list of fold numbers to use if use_specific_folds is True.
    folds: tuple[int, int] = [0, 1]  # Used only if use_specific_folds is True

    # Debugging
    # Flag to enable debug mode. If True, only a small subset of test files is processed.
    debug: bool = False
    # The number of test files to process in debug mode.
    debug_count: bool = 3

cfg = CFG()


print(f"Using device: {cfg.device}")
print(f"Loading taxonomy data...")
taxonomy_df = pd.read_csv(cfg.taxonomy_csv)
species_ids = taxonomy_df['primary_label'].tolist()
num_classes = len(species_ids)
print(f"Number of classes: {num_classes}")


class BirdCLEFModel(nn.Module):
    """
    PyTorch model for bird song classification using a backbone architecture 
    (like EfficientNet) followed by a classifier.

    Args:
        cfg (CFG): Configuration object containing model and data parameters.
        num_classes (int): The number of output classes (bird species).
    """

    def __init__(self, cfg: CFG, num_classes: int):
        super().__init__()
        self.cfg = cfg
        
        self.backbone = timm.create_model(
            cfg.model_name,
            pretrained=False, # Set to False as we are loading weights later
            in_chans=cfg.in_channels,
            drop_rate=0.0,    
            drop_path_rate=0.0
        )

        # Modify the classifier layer of the backbone based on its type
        if "efficientnet" in cfg.model_name:
            backbone_out = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()
        elif "resnet" in cfg.model_name:
            backbone_out = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        else:
            # For other timm models, get classifier features and reset it
            backbone_out = self.backbone.get_classifier().in_features
            self.backbone.reset_classifier(0, "")
        
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.feat_dim = backbone_out
        self.classifier = nn.Linear(backbone_out, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor (mel spectrogram), expected shape 
                              (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Output logits from the classifier, shape (batch_size, num_classes).
        """

        features = self.backbone(x)

        # Handle potential dictionary output from some backbones
        if isinstance(features, dict):
            features = features['features']

        # Apply pooling if the features are 4D (image-like)
        if len(features.shape) == 4:
            features = self.pooling(features)
            features = features.view(features.size(0), -1) # Flatten the features
        
        logits = self.classifier(features)
        return logits


def audio2melspec(audio_data: np.ndarray, cfg: CFG) -> np.ndarray:
    """Convert audio data to mel spectrogram

    Args:
        audio_data (np.ndarray): The input audio data as a NumPy array.
        cfg (CFG): Configuration object containing mel spectrogram parameters
             like FS, N_FFT, HOP_LENGTH, N_MELS, FMIN, FMAX.

    Returns:
        np.ndarray: The normalized mel spectrogram as a NumPy array.
    """

    # Handle potential NaN values in the audio data
    if np.isnan(audio_data).any():
        mean_signal = np.nanmean(audio_data)
        audio_data = np.nan_to_num(audio_data, nan=mean_signal)

    # Compute the mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=audio_data,
        sr=cfg.FS,
        n_fft=cfg.N_FFT,
        hop_length=cfg.HOP_LENGTH,
        n_mels=cfg.N_MELS,
        fmin=cfg.FMIN,
        fmax=cfg.FMAX,
        power=2.0 # Use power=2.0 for the power spectrogram
    )

    # Convert power spectrogram to decibels (dB)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    # Normalize the dB mel spectrogram to a 0-1 range
    mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)
    
    return mel_spec_norm


def process_audio_segment(audio_data: np.ndarray, cfg: CFG) -> np.ndarray:
    """Process audio segment to get mel spectrogram

    Pads the audio segment if it's shorter than the window size, converts it
    to a mel spectrogram using audio2melspec, and resizes the spectrogram
    to the target shape specified in the config.

    Args:
        audio_data (np.ndarray): The input audio data segment as a NumPy array.
        cfg (CFG): Configuration object containing audio and mel spectrogram parameters
             like FS, WINDOW_SIZE, TARGET_SHAPE.

    Returns:
        np.ndarray: The processed and resized mel spectrogram as a NumPy array 
                    of type np.float32.
    """

    # Pad the audio data if its length is less than the required window size
    if len(audio_data) < cfg.FS * cfg.WINDOW_SIZE:
        audio_data = np.pad(audio_data, 
                          (0, cfg.FS * cfg.WINDOW_SIZE - len(audio_data)), 
                          mode='constant')

    # Convert the audio data segment to a mel spectrogram
    mel_spec = audio2melspec(audio_data, cfg)
    
    # Resize the mel spectrogram if its shape does not match the target shape
    if mel_spec.shape != cfg.TARGET_SHAPE:
        mel_spec = cv2.resize(mel_spec, cfg.TARGET_SHAPE, interpolation=cv2.INTER_LINEAR)

    # Return the processed mel spectrogram, ensuring it's of type float32
    return mel_spec.astype(np.float32)


def find_model_files(cfg: CFG) -> list[str]:
    """
    Find all .pth model files in the specified model directory

    Args:
        cfg: Configuration object containing audio and mel spectrogram parameters
             like FS, WINDOW_SIZE, TARGET_SHAPE.

    Returns:
        list[str]: Paths of model files
    """

    model_files = []
    
    model_dir = Path(cfg.model_path)
    
    for path in model_dir.glob("**/*.pth"):
        model_files.append(str(path))
    
    return model_files


def load_models(cfg: CFG, num_classes: int) -> list[BirdCLEFModel]:
    """
    Load all found model files and prepare them for ensemble

    Searches for .pth model files in the directory specified by cfg.model_path.
    If use_specific_folds is True in the config, it filters the models to
    include only those from the specified folds. Each found model file is
    loaded into a BirdCLEFModel instance, moved to the configured device,
    and set to evaluation mode.

    Args:
        cfg (cfg): Configuration object containing model loading parameters
             like model_path, use_specific_folds, folds, device.
        num_classes (int): The number of output classes for the models.

    Returns:
        list[BirdCLEFModel]: A list of loaded BirdCLEFModel instances. Returns an empty list
              if no models are found or loaded successfully.
    """

    models = []
    
    model_files = find_model_files(cfg)
    
    if not model_files:
        print(f"Warning: No model files found under {cfg.model_path}!")
        return models
    
    print(f"Found a total of {len(model_files)} model files.")
    
    if cfg.use_specific_folds:
        filtered_files = []
        for fold in cfg.folds:
            # Basic check for fold number in file path string
            fold_files = [f for f in model_files if f"fold{fold}" in f]
            filtered_files.extend(fold_files)
        model_files = filtered_files
        print(f"Using {len(model_files)} model files for the specified folds ({cfg.folds}).")

    # Load models individually
    for model_path in model_files:
        try:
            print(f"Loading model: {model_path}")
            # Load the model checkpoint, mapping to the specified device
            checkpoint = torch.load(model_path, map_location=torch.device(cfg.device), weights_only=False)

            # Initialize the model architecture
            model = BirdCLEFModel(cfg, num_classes)

            # Load the state dictionary from the checkpoint
            model.load_state_dict(checkpoint['model_state_dict'])

            # Move the model to the specified device
            model = model.to(cfg.device)

            # Set the model to evaluation mode (disables dropout, batch normalization updates, etc.)
            model.eval()

            # Add the loaded model to the list
            models.append(model)
        except Exception as e:
            # Print an error message if loading fails for a specific model
            print(f"Error loading model {model_path}: {e}")

    return models


def predict_on_spectrogram(audio_path: str, models: list[BirdCLEFModel], cfg: CFG, species_ids) -> tuple[int, np.ndarray]:
    """
    Process a single audio file and predict species presence for each 5-second segment.

    Loads an audio file, divides it into segments, processes each segment
    into a mel spectrogram, applies TTA if enabled, passes the spectrogram(s)
    through the model(s), and collects the predicted probabilities for each
    species in each segment.

    Args:
        audio_path (str): The file path to the audio file (e.g., .ogg).
        models (list[BirdCLEFModel]): A list of loaded BirdCLEFModel instances for inference
                (can be a single model in a list or an ensemble).
        cfg (CFG): Configuration object containing inference parameters
             like FS, WINDOW_SIZE, use_tta, tta_count, device.
        species_ids (list[int]): A list of species IDs corresponding to the model output indices.

    Returns:
        tuple: A tuple containing two elements:
            - list: A list of row_ids (strings) for each segment.
            - list: A list of NumPy arrays, where each array contains the 
                    predicted probabilities for all species for a segment.
              Returns empty lists within the tuple if processing fails.
    """

    predictions = []
    row_ids = []
    soundscape_id = Path(audio_path).stem
    
    try:
        print(f"Processing {soundscape_id}")
        # Load the full audio file
        audio_data, _ = librosa.load(audio_path, sr=cfg.FS)

        # Calculate the total number of full 5-second segments
        total_segments = int(len(audio_data) / (cfg.FS * cfg.WINDOW_SIZE))

        # Process each segment
        for segment_idx in range(total_segments):
            start_sample = segment_idx * cfg.FS * cfg.WINDOW_SIZE
            end_sample = start_sample + cfg.FS * cfg.WINDOW_SIZE
            segment_audio = audio_data[start_sample:end_sample]

            # Determine the end time for the row_id
            end_time_sec = (segment_idx + 1) * cfg.WINDOW_SIZE
            row_id = f"{soundscape_id}_{end_time_sec}"
            row_ids.append(row_id)

            # Inference with or without TTA
            if cfg.use_tta:
                all_preds = []
                
                for tta_idx in range(cfg.tta_count):
                    # Process segment and apply TTA
                    mel_spec = process_audio_segment(segment_audio, cfg)
                    mel_spec = apply_tta(mel_spec, tta_idx)

                    # Prepare spectrogram for the model (add batch and channel dimensions)
                    mel_spec = torch.tensor(mel_spec, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                    mel_spec = mel_spec.to(cfg.device)

                    # Perform inference with single model or ensemble for this TTA variation
                    if len(models) == 1:
                        with torch.no_grad():
                            outputs = models[0](mel_spec)
                            probs = torch.sigmoid(outputs).cpu().numpy().squeeze()
                            all_preds.append(probs)
                    else: # Ensemble
                        segment_preds = []
                        for model in models:
                            with torch.no_grad():
                                outputs = model(mel_spec)
                                probs = torch.sigmoid(outputs).cpu().numpy().squeeze()
                                segment_preds.append(probs)
                        
                        avg_preds = np.mean(segment_preds, axis=0)
                        all_preds.append(avg_preds)

                # Average predictions across all TTA variations
                final_preds = np.mean(all_preds, axis=0)
            else: # No TTA
                mel_spec = process_audio_segment(segment_audio, cfg)

                # Prepare spectrogram for the model
                mel_spec = torch.tensor(mel_spec, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                mel_spec = mel_spec.to(cfg.device)

                # Perform inference with single model or ensemble
                if len(models) == 1:
                    with torch.no_grad():
                        outputs = models[0](mel_spec)
                        final_preds = torch.sigmoid(outputs).cpu().numpy().squeeze()
                else: # Ensemble
                    segment_preds = []
                    for model in models:
                        with torch.no_grad():
                            outputs = model(mel_spec)
                            probs = torch.sigmoid(outputs).cpu().numpy().squeeze()
                            segment_preds.append(probs)

                    final_preds = np.mean(segment_preds, axis=0)
                    
            predictions.append(final_preds)
            
    except Exception as e:
        # Print an error if processing the audio file fails
        print(f"Error processing {audio_path}: {e}")
    
    return row_ids, predictions


def apply_tta(spec: np.ndarray, tta_idx: int) -> np.ndarray:
    """
    Apply test-time augmentation to a mel spectrogram.

    Applies different transformations to the input spectrogram based on the 
    augmentation index. Supported transformations are original, horizontal flip
    (time shift), and vertical flip (frequency shift).

    Args:
        spec (np.ndarray): The input mel spectrogram as a NumPy array.
        tta_idx (int): The index specifying which TTA transformation to apply.
                       - 0: Original (no transformation)
                       - 1: Horizontal flip (time axis)
                       - 2: Vertical flip (frequency axis)
                       - Others: Original (no transformation)

    Returns:
        np.ndarray: The augmented mel spectrogram as a NumPy array.
    """

    if tta_idx == 0:
        # Original spectrogram
        return spec
    elif tta_idx == 1:
        # Time shift (horizontal flip) by flipping along the second axis (columns)
        return np.flip(spec, axis=1)
    elif tta_idx == 2:
        # Frequency shift (vertical flip) by flipping along the first axis (rows)
        return np.flip(spec, axis=0)
    else:
        # Default to original if index is not recognized
        return spec


def run_inference(cfg: CFG, models: list[BirdCLEFModel], species_ids: list[int]) -> tuple[int, np.ndarray]:
    """
    Run inference on all test soundscapes.

    Finds all audio files in the test soundscape directory, optionally limits
    the number of files in debug mode, and calls predict_on_spectrogram for
    each audio file to get predictions for all segments. Aggregates the
    results from all files.

    Args:
        cfg: Configuration object containing inference parameters
             like test_soundscapes, debug, debug_count.
        models: A list of loaded BirdCLEFModel instances for inference.
        species_ids: A list of species IDs used for predictions.

    Returns:
        tuple: A tuple containing two elements:
            - list: A list of all row_ids from all processed segments 
                    across all test files.
            - list: A list of NumPy arrays, where each array contains the 
                    predicted probabilities for all species for a segment. 
                    This list contains results from all segments of all files.
    """

    # Find all audio files in the test soundscapes directory
    test_files = list(Path(cfg.test_soundscapes).glob('*.ogg'))

    # Apply debug mode if enabled
    if cfg.debug:
        print(f"Debug mode enabled, using only {cfg.debug_count} files")
        test_files = test_files[:cfg.debug_count]
    
    print(f"Found {len(test_files)} test soundscapes")

    all_row_ids = []
    all_predictions = []

    # Process each test audio file using tqdm for a progress bar
    for audio_path in tqdm(test_files):
        # Get predictions for all segments of the current audio file
        row_ids, predictions = predict_on_spectrogram(str(audio_path), models, cfg, species_ids)

        # Extend the master lists with results from the current file
        all_row_ids.extend(row_ids)
        all_predictions.extend(predictions)

    # Return the aggregated row IDs and predictions
    return all_row_ids, all_predictions


def create_submission(row_ids: list[int], predictions: list[np.ndarray], species_ids: list[int], cfg: CFG) -> pd.DataFrame:
    """
    Create submission dataframe in the required format.

    Constructs a pandas DataFrame from the row IDs and predictions.
    It ensures that all species columns present in the sample submission file
    are included, adding columns with 0.0 values for any missing species.
    The columns are ordered according to the sample submission.

    Args:
        row_ids (list): A list of row_ids for each prediction segment.
        predictions (list): A list of NumPy arrays, where each array contains the
                             predicted probabilities for all species for a segment.
        species_ids (list): A list of species IDs corresponding to the order of 
                            probabilities in the predictions arrays.
        cfg: Configuration object containing the path to the sample submission CSV
             (submission_csv).

    Returns:
        pd.DataFrame: A pandas DataFrame formatted for submission, with 'row_id'
                      as the first column and species columns following.
    """

    print("Creating submission dataframe...")

    # Create a dictionary to build the DataFrame
    submission_dict = {"row_id": row_ids}

    # Add columns for each species with their predicted probabilities
    # Each 'pred' in predictions is a numpy array of probabilities corresponding to species_ids order
    for i, species in enumerate(species_ids):
        # Collect the prediction probability for the i-th species across all segments
        submission_dict[species] = [pred[i] for pred in predictions]

    # Create the initial DataFrame
    submission_df = pd.DataFrame(submission_dict)

    # Set "row_id" as the index temporarily for alignment with sample submission
    submission_df.set_index("row_id", inplace=True)

    # Read the sample submission to get the required columns and order
    sample_sub = pd.read_csv(cfg.submission_csv, index_col='row_id')

    # Check for any species columns present in the sample submission but missing in our dataframe
    missing_cols = set(sample_sub.columns) - set(submission_df.columns)
    if missing_cols:
        print(f"Warning: Missing {len(missing_cols)} species columns in submission")
        # Add missing columns with default value 0.0
        for col in missing_cols:
            submission_df[col] = 0.0

    # Reindex the submission dataframe to match the column order of the sample submission
    submission_df = submission_df[sample_sub.columns]

    # Reset the index to make 'row_id' a regular column again
    submission_df = submission_df.reset_index()
    
    return submission_df


print("Starting BirdCLEF-2025 inference...")
print(f"TTA enabled: {cfg.use_tta} (variations: {cfg.tta_count if cfg.use_tta else 0})")


models = load_models(cfg, num_classes)


if not models:
    raise Exception("No models found! Please check model paths.")


print(f"Model usage: {'Single model' if len(models) == 1 else f'Ensemble of {len(models)} models'}")


list(Path(cfg.test_soundscapes).glob("*"))


row_ids, predictions = run_inference(cfg, models, species_ids)


submission_df = create_submission(row_ids, predictions, species_ids, cfg)


# Test data is private except when submitted, so the DataFrame is empty.
submission_df.head()


submission_path = "submission.csv"
submission_df.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")




