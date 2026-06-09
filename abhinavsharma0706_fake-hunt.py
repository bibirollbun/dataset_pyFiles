# Install sentence-transformers for text embedding
!pip install -U sentence-transformers

# Imports
import os
import glob
import random
import string
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Function to fix random seeds for reproducibility
def set_seeds(seed=2025):
    os.environ['PYTHONHASHSEED'] = str(seed)        # For Python internal hash-based operations
    random.seed(seed)                               # Python built-in random seed
    np.random.seed(seed)                            # NumPy random seed
    tf.random.set_seed(seed)                        # TensorFlow random seed

# Set seeds
set_seeds(2026)


def convert_folder_to_csv(base_path: str, label_df: pd.DataFrame = None, split: str = "train") -> pd.DataFrame:
    """
    Reads text files from article folders and combines them into a dataframe.
    
    Args:
        base_path: Root directory containing 'train' or 'test' folders.
        label_df: DataFrame containing labels (optional, for train split).
        split: "train" or "test" to select folder.

    Returns:
        Pandas DataFrame with columns: id, text1, text2, (optional) real_text_id.
    """
    data = []

    # Get sorted list of article folders, e.g., article_0, article_1, ...
    folders = sorted(glob.glob(os.path.join(base_path, split, "article_*")))

    for i, folder in enumerate(folders):
        # Paths to the two text files in each article folder
        file_1_path = os.path.join(folder, "file_1.txt")
        file_2_path = os.path.join(folder, "file_2.txt")

        # Read the content of both text files
        with open(file_1_path, "r", encoding="utf-8") as f1, open(file_2_path, "r", encoding="utf-8") as f2:
            text1 = f1.read().strip()
            text2 = f2.read().strip()

        # Append label if available (training mode)
        if label_df is not None:
            real_id = label_df.iloc[i]["real_text_id"]
            data.append({"id": i, "text1": text1, "text2": text2, "real_text_id": real_id})
        else:
            data.append({"id": i, "text1": text1, "text2": text2})

    return pd.DataFrame(data)

# Load training labels CSV
label_df = pd.read_csv(r'/kaggle/input/fake-or-real-the-impostor-hunt/data/train.csv')

# Convert train and test folder data to DataFrames
train = convert_folder_to_csv('/kaggle/input/fake-or-real-the-impostor-hunt/data', label_df, "train")
test = convert_folder_to_csv('/kaggle/input/fake-or-real-the-impostor-hunt/data', None, "test")

# Save dataframes as CSV for easier reloading
train_path = "/kaggle/working/train_data.csv"
test_path  = "/kaggle/working/test_data.csv"

train.to_csv(train_path, index=False)
test.to_csv(test_path, index=False)

print(f'Train csv saved at {train_path}')
print(f'Test csv saved at {test_path}')


# Load saved train and test data CSVs
df = pd.read_csv(train_path)

# Drop any rows with missing text data, just in case
df = df.dropna(subset=["text1", "text2"]).reset_index(drop=True)

test_df = pd.read_csv(test_path)

# Convert label values: 1 -> 1 (real), 2 -> 0 (fake)
df["label"] = df["real_text_id"].apply(lambda x: 1 if x == 1 else 0)

# Show first few rows for confirmation
df.head()


# Load pre-trained SentenceTransformer model
model = SentenceTransformer("paraphrase-mpnet-base-v2")

# Generate embeddings for both text columns in train set
text1_embeddings = model.encode(df["text1"].tolist(), convert_to_numpy=True, show_progress_bar=True)
text2_embeddings = model.encode(df["text2"].tolist(), convert_to_numpy=True, show_progress_bar=True)

print(f'text1 shape : {text1_embeddings.shape}')  # e.g. (num_samples, 768)
print(f'text2 shape : {text2_embeddings.shape}')


# Concatenate features: [text1_emb, text2_emb, absolute difference]
X = np.concatenate([
    text1_embeddings,
    text2_embeddings,
    np.abs(text1_embeddings - text2_embeddings)  # This helps capture similarity/difference
], axis=1)

# Target labels
y = df["label"].values

print(f'X shape : {X.shape}')  # Should be (num_samples, 768*3)
print(f'y shape : {y.shape}')


# Stratified split to maintain label distribution
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=2025
)

print('X train shape : ', X_train.shape)
print('X val shape   : ', X_val.shape)
print('y train shape : ', y_train.shape)
print('y val shape   : ', y_val.shape)


# Use CPU explicitly (for Kaggle kernels that might default to GPU)
with tf.device('/CPU:0'):
    
    # Build simple MLP model for binary classification
    model_mlp = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(512, activation="relu", kernel_regularizer=tf.keras.regularizers.L2(0.1)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(256, activation="relu", kernel_regularizer=tf.keras.regularizers.L2(0.1)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid")  # Output probability for binary class
    ])

    model_mlp.compile(
        optimizer='adam',
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    # Early stopping callback to prevent overfitting
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=25,
        restore_best_weights=True,
        verbose=True
    )

    # Train model
    history = model_mlp.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=16,
        callbacks=[early_stop]
    )


# Encode test set text pairs using same SentenceTransformer model
test_text1_emb = model.encode(test_df["text1"].tolist(), convert_to_numpy=True, show_progress_bar=True)
test_text2_emb = model.encode(test_df["text2"].tolist(), convert_to_numpy=True, show_progress_bar=True)

# Create test features same way as train features
X_test = np.concatenate([
    test_text1_emb,
    test_text2_emb,
    np.abs(test_text1_emb - test_text2_emb)
], axis=1)

print(f'X test shape : {X_test.shape}')

# Predict with trained model on test features
with tf.device('/CPU:0'):
    test_preds = model_mlp.predict(X_test)

# Convert sigmoid outputs to class labels (threshold = 0.5)
# Recall: original labels are 1 for real, 2 for fake
test_df["real_text_id"] = test_preds
test_df["real_text_id"] = test_df["real_text_id"].apply(lambda x: 1 if x >= 0.5 else 2)

# Prepare submission file
submission = test_df[["id", "real_text_id"]]

# Check distribution of predicted labels
print(submission['real_text_id'].value_counts())

submission.to_csv("submission.csv", index=False)


# Reduce high-dimensional features to 2D for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.figure(figsize=(6, 5))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='coolwarm', alpha=0.6)
plt.title("PCA Projection of Feature Space")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar(label='Label (0=Fake, 1=Real)')
plt.show()

