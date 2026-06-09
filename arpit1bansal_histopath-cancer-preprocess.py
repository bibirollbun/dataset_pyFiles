from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
secret_value_0 = user_secrets.get_secret("SD3.5_token")

from huggingface_hub import login
login(token=secret_value_0)


# !pip install -q tf-nightly[and-cuda]


import os
import glob
import time
import h5py
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import tensorflow as tf
from huggingface_hub import from_pretrained_keras

# Define paths
DATA_PATH = "/kaggle/input/histopathologic-cancer-detection/train/"
OUTPUT_PATH = "/kaggle/working/embeddings/"
METADATA_PATH = "/kaggle/working/metadata/"

# Create directories if they don't exist
os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(METADATA_PATH, exist_ok=True)

def load_model():
    """Load PathFoundation model from Hugging Face"""
    print("Loading PathFoundation model...")
    model = from_pretrained_keras("google/path-foundation")
    infer = model.signatures["serving_default"]
    print("Model loaded!")
    return infer

def process_image(image_path, infer_function):
    """Process a single image and get embedding"""
    try:
        # Open and prepare image
        img = Image.open(image_path).convert('RGB')
        
        # Resize to 224x224 if needed
        if img.size != (224, 224):
            img = img.resize((224, 224))
        
        # Convert to tensor and normalize
        tensor = tf.cast(tf.expand_dims(np.array(img), axis=0), tf.float32) / 255.0
        
        # Get embedding
        embeddings = infer_function(tf.constant(tensor))
        embedding_vector = embeddings['output_0'].numpy().flatten()
        
        return embedding_vector
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def process_in_batches(file_list, batch_size=1000, model_infer=None, labels_df=None):
    """Process files in batches and save embeddings to HDF5"""
    if model_infer is None:
        model_infer = load_model()
    
    # Create labels dictionary for faster lookup
    labels_dict = {}
    if labels_df is not None:
        labels_dict = dict(zip(labels_df['id'], labels_df['label']))
    
    total_batches = (len(file_list) + batch_size - 1) // batch_size
    
    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(file_list))
        batch_files = file_list[start_idx:end_idx]
        
        batch_name = f"batch_{batch_idx:05d}"
        h5_path = os.path.join(OUTPUT_PATH, f"{batch_name}.h5")
        
        # Create metadata for this batch
        metadata = {
            'file_id': [],
            'file_path': [],
            'embedding_batch': [],
            'embedding_index': [],
            'label': []
        }
        
        # Process batch
        with h5py.File(h5_path, 'w') as h5f:
            embeddings_dataset = h5f.create_dataset(
                'embeddings', 
                shape=(len(batch_files), 384),
                dtype='float32'
            )
            
            for i, file_path in enumerate(tqdm(batch_files, desc=f"Processing batch {batch_idx+1}/{total_batches}")):
                file_id = os.path.basename(file_path).split('.')[0]
                embedding = process_image(file_path, model_infer)
                
                if embedding is not None:
                    # Save embedding to HDF5
                    embeddings_dataset[i] = embedding
                    
                    # Store metadata
                    metadata['file_id'].append(file_id)
                    metadata['file_path'].append(file_path)
                    metadata['embedding_batch'].append(batch_name)
                    metadata['embedding_index'].append(i)
                    # Add label if available
                    if file_id in labels_dict:
                        metadata['label'].append(labels_dict[file_id])
                    else:
                        metadata['label'].append(None)
        
        # Save metadata for this batch
        metadata_df = pd.DataFrame(metadata)
        metadata_df.to_parquet(os.path.join(METADATA_PATH, f"{batch_name}_metadata.parquet"))
        
        print(f"Completed batch {batch_idx+1}/{total_batches}")

def main():
    # Get list of all .tif files
    print("Finding all .tif files...")
    tif_files = glob.glob(os.path.join(DATA_PATH, "*.tif"))
    print(f"Found {len(tif_files)} .tif files")
    
    # Load labels if available
    labels_df = None
    labels_path = "/kaggle/input/histopathologic-cancer-detection/train_labels.csv"
    if os.path.exists(labels_path):
        print("Loading cancer labels...")
        labels_df = pd.read_csv(labels_path)
        print(f"Loaded {len(labels_df)} labels")
    
    # Load model once
    model_infer = load_model()
    
    # Process in batches
    start_time = time.time()
    process_in_batches(tif_files, batch_size=1000, model_infer=model_infer, labels_df=labels_df)
    end_time = time.time()
    
    print(f"Processing completed in {(end_time - start_time)/60:.2f} minutes")
    
    # Create a master metadata file linking all batches
    print("Creating master metadata file...")
    metadata_files = glob.glob(os.path.join(METADATA_PATH, "*_metadata.parquet"))
    if metadata_files:
        master_metadata = pd.concat([pd.read_parquet(f) for f in metadata_files])
        master_metadata.to_parquet(os.path.join(METADATA_PATH, "master_metadata.parquet"))
        print(f"Master metadata saved with {len(master_metadata)} entries")
    
    print("Done!")

if __name__ == "__main__":
    main()




