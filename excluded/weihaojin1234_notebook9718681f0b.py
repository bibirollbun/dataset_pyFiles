import h5py
import tensorflow as tf
from PIL import Image
import io

def generate_dataset(hdf5_file, batch_size):
    # Get keys
    with h5py.File(hdf5_file, 'r') as file:
        keys = list(file.keys())
    
    # Get not processed images from hdf5
    img_dataset = tf.data.Dataset.from_generator(
        image_generator,
        args=[hdf5_file],
        output_signature=tf.TensorSpec(shape=(None, None, 3), dtype=tf.uint8)
    )
    
    # Preprocess the images within the dataset pipeline
    #img_dataset = img_dataset.map(preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        
    # Batch the data
    img_dataset = img_dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

    #return combined_dataset

    return img_dataset

def image_generator(hdf5_file):
    with h5py.File(hdf5_file, 'r') as file:
        # List all keys in the HDF5 file
        idx = list(file.keys())
        
        # Iterate over the keys
        for key in idx:
            # Access the dataset and convert it to bytes
            image_data = file.get(key)[()]
            
            # Convert bytes to image using PIL
            image = Image.open(io.BytesIO(image_data))
            
            # Convert image to numpy format
            image_array = np.array(image)
            
            img_resized = tf.image.resize(image_array, (224, 224))
    
            #img_resized = np.expand_dims(img_resized, axis = 0)    

            #yield image_array
            
            yield img_resized


import torch
import torch.nn as nn
import timm
import numpy as np
import tensorflow as tf
from torchvision import transforms

# Load model
model = timm.create_model('vit_small_patch16_224', pretrained=False, num_classes=2)
checkpoint = torch.load('/kaggle/input/vit_small/transformers/default/1/checkpoint_epoch_1.pt', map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Load TF dataset
hdf5_file = '/kaggle/input/isic-2024-challenge/test-image.hdf5'
batch_size = 32
tf_dataset = generate_dataset(hdf5_file, batch_size)

# Normalize transformation to match ViT requirements
normalize = transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)

# Run inference
predictions = []
for batch in tf_dataset:
    batch_np = batch.numpy().astype(np.uint8)  # shape (B, 224, 224, 3)
    batch_np = np.transpose(batch_np, (0, 3, 1, 2))  # to (B, 3, 224, 224)

    # Convert to torch.Tensor and normalize
    batch_tensor = torch.tensor(batch_np / 255.0, dtype=torch.float32)
    for i in range(batch_tensor.size(0)):
        batch_tensor[i] = normalize(batch_tensor[i])
    
    with torch.no_grad():
        logits = model(batch_tensor)
        probs = torch.softmax(logits, dim=1)[:, 1]  # malignant class
        predictions.extend(probs.numpy())



import pandas as pd

with h5py.File(hdf5_file, 'r') as file:
        keys = list(file.keys())

submission_df_avg = pd.DataFrame({
    'isic_id': keys,
    'target': predictions
})

submission_df_avg.to_csv("submission.csv", index=False)


print(submission_df_avg)

