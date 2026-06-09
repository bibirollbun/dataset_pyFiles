import os  # Import the OS module for working with file paths and directories

# Define the base path for the dataset
base_path = "/kaggle/input/hms-harmful-brain-activity-classification"

# Define the directory for storing spectrograms
SPEC_DIR = "/tmp/dataset/hms-hbac"

# Create directories for training and testing spectrograms if they don't exist
os.makedirs(SPEC_DIR + '/train_spectrograms', exist_ok=True)
os.makedirs(SPEC_DIR + '/test_spectrograms', exist_ok=True)


import pandas as pd 
df =  pd.read_csv(f"{base_path}/train.csv")
df.head()


df["expert_consensus"].unique()


labels = {"Seizure": 0 ,
          "GPD" : 1 ,
          "LRDA" : 2 ,
          "Other" : 3 ,
          "GRDA" : 4 ,
          "LPD" : 5 }


# Create the EEG file path for each row in the DataFrame based on `eeg_id`
df["eeg_path"] = f"{base_path}/train_eegs/" + df["eeg_id"].astype(str) + ".parquet"

# Create the spectrogram file path for each row in the DataFrame based on `spectrogram_id`
df["spec_path"] = f"{base_path}/train_spectrograms/" + df["spectrogram_id"].astype(str) + ".parquet"

# Create an alternative spectrogram file path with .npy format for each row
df["spec_path2"] = f"{SPEC_DIR}/train_spectrograms/" + df["spectrogram_id"].astype(str) + ".npy"

# Copy the values from the `expert_consensus` column to `class_name`
df["class_name"] = df["expert_consensus"].copy()

# Map class names to integer labels using the `labels` dictionary
df["class_label"] = df.expert_consensus.map(labels).astype("int64")

# Display the first few rows of the DataFrame
df.head()



test_df = pd.read_csv(f"{base_path}/test.csv")
test_df.head()


test_df["eeg_path"] = f"{base_path}/test_eegs/"+test_df["eeg_id"].astype(str)+".parquet"
test_df["spec_path"] = f"{base_path}/test_spectrograms/"+test_df["spectrogram_id"].astype(str)+".parquet"
test_df["spec_path2"] = f"{SPEC_DIR}/test_spectrograms/"+test_df["spectrogram_id"].astype(str)+".npy"
test_df.head()


def process_spec(spec_id, split="train"):
    # Define the path to the spectrogram file (parquet format)
    spec_path = f"{base_path}/{split}_spectrograms/{spec_id}.parquet"
    
    # Read the spectrogram data from the parquet file
    spec = pd.read_parquet(spec_path)
    
    # Fill any missing values with 0, remove the first column, and transpose the matrix
    spec = spec.fillna(0).values[:, 1:].T  
    
    # Convert the spectrogram data type to float32
    spec = spec.astype("float32")
    
    # Save the processed spectrogram as a .npy file in the specified directory
    np.save(f"{SPEC_DIR}/{split}_spectrograms/{spec_id}.npy", spec)



import numpy as np
from tqdm.notebook import tqdm  # Import tqdm for progress bar display in Jupyter Notebook
import joblib  # Import joblib for parallel processing

# Get unique spectrogram IDs from the training DataFrame
spec_ids = df["spectrogram_id"].unique()

# Process each spectrogram ID in parallel for the training dataset
_ = joblib.Parallel(n_jobs=-1, backend="loky")(
    joblib.delayed(process_spec)(spec_id, "train")  # Call `process_spec` for each spectrogram ID
    for spec_id in tqdm(spec_ids, total=len(spec_ids))  # Show progress bar
)

# Get unique spectrogram IDs from the test DataFrame
test_spec_ids = test_df["spectrogram_id"].unique()

# Process each spectrogram ID in parallel for the test dataset
_ = joblib.Parallel(n_jobs=-1, backend="loky")(
    joblib.delayed(process_spec)(spec_id, "test")  # Call `process_spec` for each spectrogram ID
    for spec_id in tqdm(test_spec_ids, total=len(test_spec_ids))  # Show progress bar
)


import tensorflow as tf
import keras_cv  # Import KerasCV for advanced augmentation techniques

def build_augmenter(dim=[400,300]):
    # Define a list of augmentation layers
    augmenters = [
        keras_cv.layers.MixUp(alpha=2.0),  # MixUp augmentation to blend images
        keras_cv.layers.RandomCutout(height_factor=(1.0, 1.0), 
                                     width_factor=(0.06, 0.1)),  # Horizontal cutout
        keras_cv.layers.RandomCutout(height_factor=(0.06, 0.1), 
                                     width_factor=(1.0, 1.0)),  # Vertical cutout
    ]
    
    def augment(img, label):
        # Wrap image and label into a dictionary for augmentation layers
        data = {"images": img, "labels": label}
        
        # Apply each augmentation with a 50% probability
        for augmenter in augmenters:
            if tf.random.uniform([]) < 0.5:
                data = augmenter(data, training=True)
        
        # Return the augmented image and label
        return data["images"], data["labels"]
    
    return augment


def build_decoder(with_labels=True, target_size=[400, 300], dtype=32):
    def decode_signal(path, offset=None):
        # Read the raw binary file
        file_bytes = tf.io.read_file(path)
        
        # Decode the raw bytes into a float32 tensor
        sig = tf.io.decode_raw(file_bytes, tf.float32)
        
        # Skip the first 1024/dtype elements (likely metadata or header)
        sig = sig[1024 // dtype:]
        
        # Reshape the signal into a 2D array of shape [400, -1]
        sig = tf.reshape(sig, [400, -1])

        # Apply offset-based cropping if needed
        if offset is not None:
            offset = offset // 2  # Adjust offset
            sig = sig[:, offset:offset + 300]  # Crop the signal
            
            # Calculate padding size if cropped signal is smaller than 300 columns
            pad_size = tf.math.maximum(0, 300 - tf.shape(sig)[1])
            
            # Apply padding to maintain the shape [400, 300]
            sig = tf.pad(sig, [[0, 0], [0, pad_size]])
            sig = tf.reshape(sig, [400, 300])

        # Clip values to avoid log(0) issues, using an exponential range
        sig = tf.clip_by_value(sig, tf.math.exp(-4.0), tf.math.exp(8.0))
        
        # Apply logarithm to the signal values
        sig = tf.math.log(sig)

        # Normalize the signal (zero mean, unit variance)
        sig -= tf.math.reduce_mean(sig)
        sig /= tf.math.reduce_std(sig) + 1e-6  # Avoid division by zero

        # Convert the signal into a 3-channel format (for CNN input compatibility)
        sig = tf.tile(sig[..., None], [1, 1, 3])  

        return sig

    def decode_label(label):
        # Convert label into a one-hot encoded vector with 6 classes
        label = tf.one_hot(label, 6)
        
        # Cast label to float32
        label = tf.cast(label, tf.float32)
        
        # Ensure the shape is [6]
        label = tf.reshape(label, [6])
        
        return label

    def decode_with_labels(path, offset=None, label=None):
        # Decode signal and label together
        sig = decode_signal(path, offset)
        label = decode_label(label)
        return sig, label

    # Return the appropriate function based on `with_labels`
    return decode_with_labels if with_labels else decode_signal



def build_dataset(paths, offsets=None, labels=None, batch_size=32, cache=True,
                  decode_fn=None, augment_fn=None,
                  augment=False, repeat=True, shuffle=1024, 
                  cache_dir="", drop_remainder=False):
    # Create cache directory if caching is enabled
    if cache_dir != "" and cache is True:
        os.makedirs(cache_dir, exist_ok=True)
    
    # If no decoding function is provided, use the default decoder
    if decode_fn is None:
        decode_fn = build_decoder(labels is not None)
    
    # If no augmentation function is provided, use the default augmenter
    if augment_fn is None:
        augment_fn = build_augmenter()
    
    AUTO = tf.data.experimental.AUTOTUNE  # Optimize performance with auto-tuning

    # Create dataset slices: (paths, offsets) for unsupervised, (paths, offsets, labels) for supervised
    slices = (paths, offsets) if labels is None else (paths, offsets, labels)
    
    # Load data from tensor slices
    ds = tf.data.Dataset.from_tensor_slices(slices)
    
    # Decode the signals and labels
    ds = ds.map(decode_fn, num_parallel_calls=AUTO)
    
    # Cache dataset if enabled
    ds = ds.cache(cache_dir) if cache else ds
    
    # Repeat dataset indefinitely if repeat is True
    ds = ds.repeat() if repeat else ds
    
    # Shuffle dataset if shuffle is enabled
    if shuffle: 
        ds = ds.shuffle(shuffle, seed=42)  # Ensure reproducibility
        opt = tf.data.Options()
        opt.experimental_deterministic = False  # Improve performance
        ds = ds.with_options(opt)
    
    # Batch the dataset
    ds = ds.batch(batch_size, drop_remainder=drop_remainder)
    
    # Apply data augmentation if enabled
    ds = ds.map(augment_fn, num_parallel_calls=AUTO) if augment else ds
    
    # Prefetch data for better performance
    ds = ds.prefetch(AUTO)
    
    return ds


from sklearn.model_selection import StratifiedGroupKFold

# Initialize StratifiedGroupKFold with 5 splits, shuffling enabled, and a fixed random state for reproducibility
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

# Initialize a new column in the dataframe to store fold assignments
df["fold"] = -1

# Reset index to ensure proper indexing before splitting
df.reset_index(drop=True, inplace=True)

# Perform stratified group k-fold splitting
for fold, (train_idx, valid_idx) in enumerate(
    sgkf.split(df, y=df["class_label"], groups=df["patient_id"])
):
    # Assign the fold number to the validation set
    df.loc[valid_idx, "fold"] = fold

# Display the count of EEG samples per fold and class
df.groupby(["fold", "class_name"])[["eeg_id"]].count().T


sample_df = df.groupby("spectrogram_id").head(1).reset_index(drop=True)
sample_df.shape


len(sample_df["spectrogram_id"].unique())


# Split the dataset into training and validation sets based on the assigned fold values
train_df = sample_df[sample_df.fold != 0]  # Use all folds except fold 0 for training
valid_df = sample_df[sample_df.fold == 0]  # Use fold 0 as the validation set

# Print the number of samples in the training and validation sets
print(f"# Num Train: {len(train_df)} | Num Valid: {len(valid_df)}")


train_df.head()


# Extract paths, offsets, and labels for the training dataset
train_paths = train_df["spec_path2"].values  # File paths to the spectrograms
train_offsets = train_df["spectrogram_label_offset_seconds"].values.astype(int)  # Time offsets for labels
train_labels = train_df["class_label"].values  # Class labels

# Build the training dataset using the custom function
train_ds = build_dataset(
    train_paths,           # Paths to spectrogram files
    train_offsets,         # Time offsets for segmentation
    train_labels,          # Class labels
    batch_size=64,         # Number of samples per batch
    repeat=True,           # Repeat dataset for continuous training
    shuffle=True,          # Shuffle the dataset to improve generalization
    augment=True,          # Apply data augmentation
    cache=True             # Cache data for faster access
)


train_ds


valid_df.head()


valid_paths = valid_df["spec_path2"].values
valid_offsets = valid_df["spectrogram_label_offset_seconds"].values.astype(int)
valid_labels = valid_df["class_label"].values
valid_ds = build_dataset(valid_paths, valid_offsets, valid_labels, batch_size=64,
                         repeat=False, shuffle=False, augment=False, cache=True)


valid_ds


model = keras_cv.models.ImageClassifier.from_preset(
    "efficientnetv2_b2_imagenet", num_classes=6
)


model.compile(
    optimizer="adam",
    loss = "categorical_crossentropy",
    metrics = ["accuracy"]
)


model.summary()


from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=4)


history = model.fit(
    train_ds, 
    epochs=6,
    callbacks=early_stopping, 
    steps_per_epoch=len(train_df)//64,
    validation_data=valid_ds
)


import matplotlib.pyplot as plt 

plt.plot(history.history["accuracy"],label="Train Accuracy")
plt.plot(history.history["val_accuracy"],label="Validation Accuracy")
plt.legend()
plt.show()


def prediction(arr):
    # Generate predictions using the trained model
    pred = model.predict(arr)
    
    # Find the class label with the highest probability
    for k, v in labels.items():
        if np.argmax(pred) == v:
            print(k)  # Print the corresponding class name



random_arr = np.random.rand(1,300,300,3)


prediction(random_arr)


test_paths = test_df["spec_path2"].values
test_ds = build_dataset(test_paths, batch_size=min(64, len(test_df)),
                         repeat=False, shuffle=False, cache=False, augment=False)


test_ds


prediction(test_ds)

