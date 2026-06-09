# Install required packages with version pinning for reproducibility
!pip install git+https://github.com/WildlifeDatasets/wildlife-datasets@develop --quiet
!pip install git+https://github.com/WildlifeDatasets/wildlife-tools@main --quiet
!pip install timm==0.9.10 --quiet
!pip install torchvision==0.16.0 --quiet


import os
import numpy as np
import pandas as pd
import timm
import torch
import torchvision.transforms as T
from wildlife_datasets.datasets import AnimalCLEF2025
from wildlife_tools.features import DeepFeatures
from wildlife_tools.similarity import CosineSimilarity
from wildlife_tools.similarity.wildfusion import SimilarityPipeline, WildFusion
from wildlife_tools.similarity.pairwise.lightglue import MatchLightGlue
from wildlife_tools.features.local import AlikedExtractor
from wildlife_tools.similarity.calibration import IsotonicCalibration
from sklearn.metrics import pairwise_distances
from tqdm import tqdm

def create_sample_submission(dataset_query, predictions, file_name='sample_submission.csv'):
    """Create submission file with proper formatting."""
    df = pd.DataFrame({
        'image_id': dataset_query.metadata['image_id'],
        'identity': predictions
    })
    df.to_csv(file_name, index=False)
    print(f"Submission file saved as {file_name}")


# Configuration parameters
class Config:
    ROOT = '/kaggle/input/animal-clef-2025'
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    BATCH_SIZE = 32  # Increased batch size for better GPU utilization
    IMAGE_SIZE = 384  # Standard size for MegaDescriptor
    CALIBRATION_SAMPLES = 200  # Increased calibration samples for better accuracy
    
    # Model configurations
    GLOBAL_FEATURE_MODEL = 'hf-hub:BVRA/MegaDescriptor-L-384'
    LOCAL_FEATURE_MODEL = 'aliked'  # Using ALIKED for local features
    
    # Threshold parameters
    INITIAL_THRESHOLD = 0.35
    THRESHOLD_STEP = 0.05
    THRESHOLD_RANGE = [0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]


# Image transformations
transform = T.Compose([
    T.Resize([Config.IMAGE_SIZE, Config.IMAGE_SIZE]),
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])

transform_aliked = T.Compose([
    T.Resize([512, 512]),  # Optimal size for ALIKED features
    T.ToTensor()
])


# Load dataset with progress indication
print("Loading datasets...")
dataset = AnimalCLEF2025(Config.ROOT, load_label=True)
dataset_database = dataset.get_subset(dataset.metadata['split'] == 'database')
dataset_query = dataset.get_subset(dataset.metadata['split'] == 'query')

# Create calibration dataset (stratified sample)
print("Creating calibration dataset...")
calibration_indices = dataset_database.metadata.sample(
    n=min(Config.CALIBRATION_SAMPLES, len(dataset_database)),
    stratify=dataset_database.metadata['class_id']
).index
dataset_calibration = AnimalCLEF2025(Config.ROOT, df=dataset_database.metadata.iloc[calibration_indices], load_label=True)

n_query = len(dataset_query)
print(f"Loaded {len(dataset_database)} database images and {n_query} query images")
print(f"Using {len(dataset_calibration)} samples for calibration")


# Initialize models with error handling
print("Initializing models...")
try:
    # Global feature model
    model = timm.create_model(
        Config.GLOBAL_FEATURE_MODEL, 
        num_classes=0, 
        pretrained=True
    ).to(Config.DEVICE).eval()
    
    # Local feature pipeline
    matcher_aliked = SimilarityPipeline(
        matcher=MatchLightGlue(
            features=Config.LOCAL_FEATURE_MODEL,
            device=Config.DEVICE,
            batch_size=Config.BATCH_SIZE
        ),
        extractor=AlikedExtractor(),
        transform=transform_aliked,
        calibration=IsotonicCalibration()
    )
    
    # Global feature pipeline
    matcher_mega = SimilarityPipeline(
        matcher=CosineSimilarity(),
        extractor=DeepFeatures(
            model=model,
            device=Config.DEVICE,
            batch_size=Config.BATCH_SIZE
        ),
        transform=transform,
        calibration=IsotonicCalibration()
    )
    
    print("Models initialized successfully")
except Exception as e:
    print(f"Error initializing models: {str(e)}")
    raise


# Calibrate WildFusion with progress tracking
print("Calibrating WildFusion...")
try:
    wildfusion = WildFusion(
        calibrated_pipelines=[matcher_aliked, matcher_mega],
        priority_pipeline=matcher_mega
    )
    
    # Fit calibration with tqdm progress bar
    with tqdm(total=len(dataset_calibration)**2, desc="Calibration Progress") as pbar:
        wildfusion.fit_calibration(
            dataset_calibration, 
            dataset_calibration,
            update_fn=lambda x: pbar.update(x)
        )
    
    print("Calibration completed successfully")
except Exception as e:
    print(f"Error during calibration: {str(e)}")
    raise


# Compute similarity with batch processing and memory optimization
print("Computing similarities...")
try:
    # Process in batches to reduce memory usage
    batch_size = 50
    similarity = np.zeros((n_query, len(dataset_database)))
    
    for i in tqdm(range(0, n_query, batch_size), desc="Processing query images"):
        batch_end = min(i + batch_size, n_query)
        batch_query = dataset.get_subset(dataset_query.metadata.index[i:batch_end])
        similarity[i:batch_end] = wildfusion(
            batch_query, 
            dataset_database, 
            B=batch_size
        )
    
    print("Similarity computation completed")
except Exception as e:
    print(f"Error computing similarities: {str(e)}")
    raise


# Generate predictions with dynamic thresholding
print("Generating predictions...")
labels = dataset_database.labels_string
pred_idx = similarity.argmax(axis=1)
pred_scores = similarity[np.arange(n_query), pred_idx]

# Generate submissions for multiple thresholds
for threshold in Config.THRESHOLD_RANGE:
    predictions = labels[pred_idx].copy()
    predictions[pred_scores < threshold] = 'new_individual'
    
    # Calculate stats
    new_individual_ratio = (predictions == 'new_individual').mean()
    print(f"Threshold {threshold:.2f}: {new_individual_ratio*100:.1f}% new individuals")
    
    create_sample_submission(
        dataset_query, 
        predictions, 
        file_name=f'sample_submission_{int(threshold*100)}.csv'
    )


# Generate optimal threshold submission (optional)
# You can analyze validation results to choose the best threshold
optimal_threshold = Config.INITIAL_THRESHOLD
predictions = labels[pred_idx].copy()
predictions[pred_scores < optimal_threshold] = 'new_individual'
create_sample_submission(dataset_query, predictions, file_name='sample_submission_optimal.csv')

print("All submissions generated successfully")

