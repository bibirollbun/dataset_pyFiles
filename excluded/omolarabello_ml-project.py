# Kaggle notebook: Facial Keypoints Detection - baseline (Keras)
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers
import gc
print("TF version:", tf.__version__)



import os
os.listdir("/kaggle/input")



DATA_DIR = pd.read_csv('/kaggle/input/facial-keypoints-detection/IdLookupTable.csv')


import os
os.listdir("/kaggle/input")



DATA_DIR = "/kaggle/input/facial-keypoints-detection"  # ğŸ‘ˆ replace with the exact name printed
print(os.listdir(DATA_DIR))



import os
import zipfile

DATA_DIR = "/kaggle/input/facial-keypoints-detection"

# Unzip training and test files if not already extracted
with zipfile.ZipFile(os.path.join(DATA_DIR, "training.zip"), "r") as z:
    z.extractall("/kaggle/working")

with zipfile.ZipFile(os.path.join(DATA_DIR, "test.zip"), "r") as z:
    z.extractall("/kaggle/working")

print(os.listdir("/kaggle/working"))



train_df = pd.read_csv("/kaggle/working/training.csv")
test_df = pd.read_csv("/kaggle/working/test.csv")
lookup_df = pd.read_csv(os.path.join(DATA_DIR, "IdLookupTable.csv"))  # still inside the input folder

print(train_df.shape, test_df.shape, lookup_df.shape)
train_df.head()



# Load the data
train_df = pd.read_csv("/kaggle/working/training.csv")
test_df = pd.read_csv("/kaggle/working/test.csv")
lookup_df = pd.read_csv(os.path.join(DATA_DIR, "IdLookupTable.csv"))  # this one is not zipped


# Display info
print("\nData loaded successfully âœ…")
print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")
print(f"Lookup shape: {lookup_df.shape}")




# Helper function to convert pixel strings to 96x96 grayscale images
def parse_images(df, img_col="Image", size=96):
    """
    Converts the 'Image' column of pixel strings into a NumPy array
    with shape (num_samples, 96, 96, 1).
    """
    images = df[img_col].str.split().apply(lambda arr: np.array(arr, dtype=np.float32))
    X = np.stack(images.values)
    X = X.reshape(-1, size, size, 1) / 255.0  # normalize pixel values
    return X

# Handle missing values in the keypoint columns
# We'll use simple forward + backward fill for this baseline
train_df_filled = train_df.copy()
train_df_filled.fillna(method='ffill', inplace=True)
train_df_filled.fillna(method='bfill', inplace=True)


# Separate features (images) and labels (keypoints)
target_cols = train_df_filled.columns.drop("Image")
y = train_df_filled[target_cols].values.astype(np.float32)

# Normalize keypoint coordinates from [0,96] to [0,1] for training stability
y = y / 96.0


# 1ï¸�âƒ£ Unzip training.zip and test.zip
DATA_DIR = "/kaggle/input/facial-keypoints-detection"

with zipfile.ZipFile(os.path.join(DATA_DIR, "training.zip"), 'r') as zip_ref:
    zip_ref.extractall("/kaggle/working/training")

with zipfile.ZipFile(os.path.join(DATA_DIR, "test.zip"), 'r') as zip_ref:
    zip_ref.extractall("/kaggle/working/test")

# 2ï¸�âƒ£ Load the extracted CSV
train_df = pd.read_csv("/kaggle/working/training/training.csv")
print("âœ… training.csv loaded!")
print(train_df.shape)
print(train_df.columns)
print(train_df.head(2))



import numpy as np

# Function to safely convert image strings to arrays
def safe_fromstring(x):
    try:
        arr = np.fromstring(x, sep=' ')
        if arr.size == 96*96:
            return arr
    except:
        pass
    return np.nan

# 1ï¸�âƒ£ Drop rows with missing keypoints
data_clean = train_df.dropna()

# 2ï¸�âƒ£ Convert images
data_clean['Image'] = data_clean['Image'].apply(safe_fromstring)

# 3ï¸�âƒ£ Drop failed conversions
data_clean = data_clean.dropna(subset=['Image'])

print("âœ… Remaining valid rows:", len(data_clean))

# 4ï¸�âƒ£ Create X and y arrays
X = np.vstack(data_clean['Image'].values).reshape(-1, 96, 96, 1) / 255.0
y = data_clean.drop('Image', axis=1).values / 96.0

print("X shape:", X.shape)
print("y shape:", y.shape)



import tensorflow as tf
from tensorflow.keras import layers, models

# Split train/validation data
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

print("Train:", X_train.shape, "Validation:", X_val.shape)

# 1ï¸�âƒ£ Define the CNN model
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(96,96,1)),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Flatten(),
    layers.Dense(500, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(500, activation='relu'),
    layers.Dense(30)  # 15 keypoints * 2 (x, y)
])

# 2ï¸�âƒ£ Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# 3ï¸�âƒ£ Train the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=64,
    verbose=1
)



import matplotlib.pyplot as plt

# Get predictions on validation set
preds = model.predict(X_val)

# De-normalize predictions and true labels (back to 0â€“96 range)
preds = preds * 96
y_val_true = y_val * 96

# Plot a few random samples
num_samples = 5
plt.figure(figsize=(12, 6))

for i in range(num_samples):
    ax = plt.subplot(1, num_samples, i+1)
    img = X_val[i].reshape(96, 96)
    ax.imshow(img, cmap='gray')
    ax.scatter(preds[i][0::2], preds[i][1::2], c='r', s=20, label='Predicted')
    ax.scatter(y_val_true[i][0::2], y_val_true[i][1::2], c='g', s=15, label='True')
    ax.axis('off')

plt.suptitle("Facial Keypoint Predictions (Red = Predicted, Green = True)", fontsize=14)
plt.show()



import zipfile

# Unzip test.zip (already exists in your dataset folder)
DATA_DIR = "/kaggle/input/facial-keypoints-detection"

with zipfile.ZipFile(os.path.join(DATA_DIR, "test.zip"), 'r') as zip_ref:
    zip_ref.extractall("/kaggle/working/test")

test_df = pd.read_csv("/kaggle/working/test/test.csv")
print("âœ… test.csv loaded!")
print(test_df.shape)
print(test_df.head(2))



def safe_fromstring(x):
    try:
        arr = np.fromstring(x, sep=' ')
        if arr.size == 96*96:
            return arr
    except:
        pass
    return np.nan

# Convert test images
test_df['Image'] = test_df['Image'].apply(safe_fromstring)
test_df = test_df.dropna(subset=['Image'])

# Create X_test
X_test = np.vstack(test_df['Image'].values).reshape(-1, 96, 96, 1) / 255.0

print("âœ… X_test shape:", X_test.shape)



# Generate predictions
preds_test = model.predict(X_test)

# Convert back to original coordinate scale
preds_test = preds_test * 96

print("âœ… Predictions shape:", preds_test.shape)



# Load the IdLookupTable
lookup_df = pd.read_csv(os.path.join(DATA_DIR, "IdLookupTable.csv"))
print("âœ… IdLookupTable loaded!")
print(lookup_df.head(5))



# Get all feature names from training set
feature_names = train_df.columns[:-1]

# Convert predictions into a DataFrame
pred_df = pd.DataFrame(preds_test, columns=feature_names)

# Merge with lookup table
submission = lookup_df.copy()
submission['Location'] = submission.apply(
    lambda row: pred_df.loc[row.ImageId - 1, row.FeatureName], axis=1
)

# Save to CSV
submission.to_csv("submission.csv", index=False)
print("ğŸ�¯ submission.csv file created successfully!")



# Get all feature names from training set
feature_names = train_df.columns[:-1]

# Convert model predictions into a DataFrame
pred_df = pd.DataFrame(preds_test, columns=feature_names)

# Create a copy of the lookup table
submission = lookup_df.copy()

# Map predictions to the lookup table
submission['Location'] = submission.apply(
    lambda row: pred_df.loc[row.ImageId - 1, row.FeatureName], axis=1
)

# Keep only the required columns: RowId and Location
final_submission = submission[['RowId', 'Location']]

# Save to CSV
final_submission.to_csv("submission.csv", index=False)
print("ğŸ�¯ submission.csv file created successfully!")


