# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


%env CUDA_LAUNCH_BLOCKING=1



!kaggle competitions download -c home-depot-product-search-relevance


pip install mlflow


import os
import zipfile
import pandas as pd
import mlflow

# Paths to the data files
input_dir = "/kaggle/input/home-depot-product-search-relevance"
output_dir = "/kaggle/working/home-depot-product-search-relevance"
zip_files = [
    "train.csv.zip",
    "test.csv.zip",
    "product_descriptions.csv.zip",
    "attributes.csv.zip"
]

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Step 1: Unzip files
print("Unzipping files...")
for zip_file in zip_files:
    zip_path = os.path.join(input_dir, zip_file)
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(output_dir)
print("Files unzipped successfully.")

# Check extracted files
print("Extracted files:")
print(os.listdir(output_dir))



# Step 2: Load and inspect data
print("Loading data...")
train_path = os.path.join(output_dir, "train.csv")
test_path = os.path.join(output_dir, "test.csv")
descriptions_path = os.path.join(output_dir, "product_descriptions.csv")
attributes_path = os.path.join(output_dir, "attributes.csv")
# # Load datasets
# train_df = pd.read_csv(train_path)
# test_df = pd.read_csv(test_path)
descriptions_df = pd.read_csv(descriptions_path, encoding="ISO-8859-1")
attributes_df = pd.read_csv(attributes_path, encoding="ISO-8859-1")
# # Inspect data
# print("Train Data:")
# print(train_df.info())
# print(train_df.head())

# print("Test Data:")
# print(test_df.info())
# print(test_df.head())

print("Product Descriptions Data:")
print(descriptions_df.info())
print(descriptions_df.head())

print("Product Attributes Data:")
print(attributes_df.info())
print(attributes_df.head())

# Step 3: Set up MLflow tracking
print("Setting up MLflow...")
mlflow.set_tracking_uri("./mlruns")
mlflow.set_experiment("home_depot_search_relevance")

with mlflow.start_run():
    mlflow.log_param("dataset_version", "home_depot_competition")
    mlflow.log_param("files_unzipped", True)
    mlflow.log_artifact(train_path, artifact_path="data")
    mlflow.log_artifact(test_path, artifact_path="data")
    mlflow.log_artifact(descriptions_path, artifact_path="data")

print("MLflow setup complete. Tracking initialized.")


# # Load datasets with error handling for problematic lines
# train_df = pd.read_csv(train_path, encoding="utf-8", on_bad_lines="skip", engine="python")
# test_df = pd.read_csv(test_path, encoding="utf-8", on_bad_lines="skip", engine="python")

# If the above fails due to encoding, try a different encoding
# Uncomment the lines below if needed
train_df = pd.read_csv(train_path, encoding="ISO-8859-1", on_bad_lines="skip", engine="python")
test_df = pd.read_csv(test_path, encoding="ISO-8859-1", on_bad_lines="skip", engine="python")




import os
import zipfile
import pandas as pd
import mlflow
import string
import numpy as np


def clean_text(text):
    """Cleans text by removing punctuation and converting to lowercase."""
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ' '.join(text.split())  # Remove extra spaces
    return text


# Clean text columns
print("Cleaning text columns...")
train_df["product_title"] = train_df["product_title"].apply(clean_text)
train_df["search_term"] = train_df["search_term"].apply(clean_text)

test_df["product_title"] = test_df["product_title"].apply(clean_text)
test_df["search_term"] = test_df["search_term"].apply(clean_text)


print("Text cleaning and character-level processing complete.")



descriptions_df["product_description"] = descriptions_df["product_description"].apply(clean_text)
attributes_df["value"] = attributes_df["value"].fillna("").apply(clean_text)

# Merge descriptions into train and test data
print("Merging product descriptions...")
train_df = pd.merge(train_df, descriptions_df, on="product_uid", how="left")
test_df = pd.merge(test_df, descriptions_df, on="product_uid", how="left")

# Aggregate attributes by product_uid
print("Aggregating attributes...")
attributes_aggregated = attributes_df.groupby("product_uid")["value"].apply(lambda x: " ".join(x)).reset_index()
attributes_aggregated.rename(columns={"value": "attributes"}, inplace=True)

# Merge aggregated attributes into train and test data
train_df = pd.merge(train_df, attributes_aggregated, on="product_uid", how="left")
test_df = pd.merge(test_df, attributes_aggregated, on="product_uid", how="left")

# Fill missing attributes with empty string
train_df["attributes"] = train_df["attributes"].fillna("")
test_df["attributes"] = test_df["attributes"].fillna("")

# # Character-level tokenization
# def char_level_tokenize(text):
#     """Converts text into a sequence of character indices without truncation."""
#     char_dict = {char: idx + 1 for idx, char in enumerate(string.ascii_lowercase + ' ')}
#     return [char_dict.get(char, 0) for char in text]





# Ensure all columns are strings before concatenation
def safe_convert_to_string(series):
    return series.fillna("").astype(str)

# Combine all text columns into one single string
all_text = " ".join(
    pd.concat([
        safe_convert_to_string(train_df["product_title"]),
        safe_convert_to_string(train_df["search_term"]),
        safe_convert_to_string(train_df["product_description"]),
        safe_convert_to_string(train_df["attributes"]),
        safe_convert_to_string(test_df["product_title"]),
        safe_convert_to_string(test_df["search_term"]),
        safe_convert_to_string(test_df["product_description"]),
        safe_convert_to_string(test_df["attributes"])
    ]).to_list()
)

# Extract unique characters
unique_chars = sorted(set(all_text))
print("Unique characters found in the dataset:")
print(unique_chars)
print(len(unique_chars))


# Step 1: Build char_dict from unique_chars
unique_chars = sorted(set(all_text))  # Assuming all_text is created from the previous step
char_dict = {char: idx + 1 for idx, char in enumerate(unique_chars)}  # Index starts at 1
print("Character dictionary created:")
print(char_dict)


def char_level_tokenize(text, max_len=50):
    """
    Converts text into a sequence of character indices using char_dict.
    Pads with 0 if the sequence is shorter than max_len (padding added only after the sequence).
    """
    sequence = [char_dict.get(char, 0) for char in text[:max_len]]
    return sequence + [0] * (max_len - len(sequence))

# Step 3: Apply the updated tokenization function
print("Tokenizing text using dynamically generated character dictionary...")

# Tokenize product_title, search_term, product_description, and attributes
max_title_len = 50
max_desc_len = 750
max_attr_len = 250
max_search_len=30


train_df["product_title_chars"] = train_df["product_title"].apply(lambda x: char_level_tokenize(x, max_title_len))
train_df["product_description_chars"] = train_df["product_description"].apply(lambda x: char_level_tokenize(x, max_desc_len))
train_df["attributes_chars"] = train_df["attributes"].apply(lambda x: char_level_tokenize(x, max_attr_len))
train_df["search_term_chars"] = train_df["search_term"].apply(lambda x: char_level_tokenize(x, max_search_len))
train_df["item_description_chars"] =  train_df["product_description_chars"] + train_df["attributes_chars"]+train_df["product_title_chars"]

test_df["product_title_chars"] = test_df["product_title"].apply(lambda x: char_level_tokenize(x, max_title_len))
test_df["product_description_chars"] = test_df["product_description"].apply(lambda x: char_level_tokenize(x, max_desc_len))
test_df["attributes_chars"] = test_df["attributes"].apply(lambda x: char_level_tokenize(x, max_attr_len))
test_df["search_term_chars"] = test_df["search_term"].apply(lambda x: char_level_tokenize(x, max_search_len))
test_df["item_description_chars"] = test_df["product_description_chars"] + test_df["attributes_chars"] + test_df["product_title_chars"]



print("Tokenization complete using dynamic character dictionary.")



train_df


import os
import pandas as pd
import mlflow
import string
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

# Split train_df into train and validation sets
train_data, val_data = train_test_split(train_df, test_size=0.2, random_state=42)

# Define Dataset
class HomeDepotDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]  # Safely access the row with .iloc
        query = torch.tensor(row['search_term_chars'], dtype=torch.long)  # Search term indices
        description = torch.tensor(row['item_description_chars'], dtype=torch.long)  # Combined text indices
        relevance = torch.tensor(row['relevance'], dtype=torch.float32)  # Relevance score
        return query, description, relevance



# Prepare DataLoaders
train_dataset = HomeDepotDataset(train_data)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=8)

val_dataset = HomeDepotDataset(val_data)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=8)


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error
import time

# Load the solution file with test labels
solution_path = "/kaggle/input/home-depot-deep-learning-solutions/solution.csv"
solution_df = pd.read_csv(solution_path)

# Filter out rows with "Ignored" in the Usage column
solution_df = solution_df[solution_df["Usage"] != "Ignored"]
# Merge test_df with solution_df to get the true relevance labels
test_df = test_df.merge(solution_df[["id", "relevance"]], on="id", how="left")
test_df = test_df.dropna(subset=["relevance"]).reset_index(drop=True)

# Use preprocessed data from previous steps
# Combine product title, description, and attributes into a single sequence
print("Training Ridge Regression model...")
start_preprocess_time = time.time()
train_df["combined_text"] = train_df["product_title_chars"].apply(lambda x: ''.join(map(str, x))) + \
                          train_df["item_description_chars"].apply(lambda x: ''.join(map(str, x)))

# Convert numerical character indices back to text for CountVectorizer
train_df["combined_text"] = train_df["combined_text"].apply(lambda x: ''.join([list(char_dict.keys())[list(char_dict.values()).index(int(c)) - 1] if int(c) > 0 else '' for c in x]))

test_df["combined_text"] = test_df["product_title_chars"].apply(lambda x: ''.join(map(str, x))) + \
                          test_df["item_description_chars"].apply(lambda x: ''.join(map(str, x)))

# Convert numerical character indices back to text for CountVectorizer
test_df["combined_text"] = test_df["combined_text"].apply(lambda x: ''.join([list(char_dict.keys())[list(char_dict.values()).index(int(c)) - 1] if int(c) > 0 else '' for c in x]))

# Step 1: Vectorize character-level sequences
print("Vectorizing character-level sequences...")
vectorizer = CountVectorizer(analyzer="char", ngram_range=(2, 7), max_features=3000)  # Larger n-grams, more features
X = vectorizer.fit_transform(train_df["combined_text"]).toarray()
y = train_df["relevance"].values

X_test = vectorizer.transform(test_df["combined_text"]).toarray()
y_test = test_df["relevance"].values

# Step 2: Train-Test Split
print("Splitting data into train and validation sets...")
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Train a Ridge Regression model
start_model_time=time.time()
ridge_regressor = Ridge(alpha=1.0)  # L2 regularization
ridge_regressor.fit(X_train, y_train)
runtime_model = time.time() - start_model_time
runtime_whole_process=time.time()-start_preprocess_time

# Step 4: Evaluate the model on the training and validation sets
y_train_pred = ridge_regressor.predict(X_train)
y_val_pred = ridge_regressor.predict(X_val)
y_test_pred = ridge_regressor.predict(X_test)

# Calculate metrics
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
train_mae = mean_absolute_error(y_train, y_train_pred)
val_mae = mean_absolute_error(y_val, y_val_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
test_mae = mean_absolute_error(y_test, y_test_pred)

# Print results
print(f"Benchmark Results:\nRuntime: {runtime_model:.2f} seconds\nTotal Process Time: {runtime_whole_process:.2f} seconds\nTrain RMSE: {train_rmse:.4f}\nValidation RMSE: {val_rmse:.4f}\nTrain MAE: {train_mae:.4f}\nValidation MAE: {val_mae:.4f}")
print(f"\nTest RMSE: {test_rmse:.4f}\nTest MAE: {test_mae:.4f}")

# Optional: Compare feature importance for interpretation
feature_importance = pd.DataFrame({
    "Feature": vectorizer.get_feature_names_out(),
    "Importance": np.abs(ridge_regressor.coef_)
}).sort_values(by="Importance", ascending=False)

print("Top 10 Character n-grams by Importance:")
print(feature_importance.head(10))



test_df





# Define CNN-based Siamese Network
import torch.nn.functional as F
import time
class CharacterSiameseLSTM(pl.LightningModule):
    def __init__(self, vocab_size=len(char_dict), embedding_dim=128, hidden_size=64, num_layers=3, dropout_prob=0.3, learning_rate=1e-3):
        super(CharacterSiameseLSTM, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_prob = dropout_prob
        self.learning_rate = learning_rate

        self.train_mae_list = []
        self.val_mae_list = []
        self.train_rmse_list = []
        self.val_rmse_list = []

        # Define embedding layer
        self.embedding = nn.Embedding(vocab_size + 1, embedding_dim, padding_idx=0)

        # Define LSTM layers
        self.lstm = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout_prob,
            bidirectional=True,
            batch_first=True
        )

        # Define fully connected layers
        self.fc = nn.Linear(self.hidden_size * 2, 1)

        self.loss = nn.SmoothL1Loss()
        self.MAE = F.l1_loss
        self.MSE = F.mse_loss

    def forward(self, input1, input2):
        # Forward pass through embedding layer
        embedded_input1 = self.embedding(input1)
        embedded_input2 = self.embedding(input2)

        # Forward pass through LSTM layers
        output1, _ = self.lstm(embedded_input1)
        output2, _ = self.lstm(embedded_input2)

        # Get the last hidden state from each LSTM
        embedding1 = output1[:, -1, :]
        embedding2 = output2[:, -1, :]

        dist = torch.abs(embedding1 - embedding2)

        # Forward pass through fully connected layers
        output = self.fc(dist)
        return output.squeeze()  # Remove the extra dimension

    def extract_features(self, input1, input2):
        # Forward pass through embedding layer
        embedded_input1 = self.embedding(input1)
        embedded_input2 = self.embedding(input2)

        # Forward pass through LSTM layers
        output1, _ = self.lstm(embedded_input1)
        output2, _ = self.lstm(embedded_input2)

        # Get the last hidden state from each LSTM
        embedding1 = output1[:, -1, :]
        embedding2 = output2[:, -1, :]

        dist = torch.abs(embedding1 - embedding2)

        # Combine all features: embedding1, embedding2, and dist
        combined_features = torch.cat([embedding1, embedding2, dist], dim=-1)
        return combined_features.detach().cpu().numpy()

    def training_step(self, batch, batch_idx):
        query, description, relevance = batch  # Unpack the tuple
        outputs = self(query, description)
        loss = self.loss(outputs, relevance)
        MAE = self.MAE(outputs, relevance)
        RMSE = torch.sqrt(self.MSE(outputs, relevance))
        # Logging
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_mae", MAE, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_rmse", RMSE, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        query, description, relevance = batch  # Unpack the tuple
        outputs = self(query, description)
        loss = self.loss(outputs, relevance)
        MAE = self.MAE(outputs, relevance)
        RMSE = torch.sqrt(self.MSE(outputs, relevance))
        # Logging
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_mae", MAE, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_rmse", RMSE, on_step=False, on_epoch=True, prog_bar=True)
        return loss


    def on_train_epoch_end(self):
        train_loss = self.trainer.callback_metrics["train_loss"].item()
        train_mae = self.trainer.callback_metrics["train_mae"].item()
        train_rmse = self.trainer.callback_metrics["train_rmse"].item()

        self.train_mae_list.append(train_mae)
        self.train_rmse_list.append(train_rmse)

        print(f"Epoch {self.current_epoch + 1}: Train MAE: {train_mae:.4f}, Train RMSE: {train_rmse:.4f}")
        mlflow.log_metric("train_loss", train_loss, step=self.current_epoch + 1)
        mlflow.log_metric("train_mae", train_mae, step=self.current_epoch + 1)
        mlflow.log_metric("train_rmse", train_rmse, step=self.current_epoch + 1)

    def on_validation_epoch_end(self):
        val_loss = self.trainer.callback_metrics["val_loss"].item()
        val_mae = self.trainer.callback_metrics["val_mae"].item()
        val_rmse = self.trainer.callback_metrics["val_rmse"].item()

        self.val_mae_list.append(val_mae)
        self.val_rmse_list.append(val_rmse)

        print(f"Epoch {self.current_epoch + 1}: Validation MAE: {val_mae:.4f}, Validation RMSE: {val_rmse:.4f}")
        mlflow.log_metric("val_loss", val_loss, step=self.current_epoch + 1)
        mlflow.log_metric("val_mae", val_mae, step=self.current_epoch + 1)
        mlflow.log_metric("val_rmse", val_rmse, step=self.current_epoch + 1)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": lr_scheduler, "monitor": "val_loss"}
        }



# Instantiate the model
model = CharacterSiameseLSTM()

# Set up MLflow logging
mlflow.set_experiment("Home Depot Relevance Training - Optimized LSTM")

with mlflow.start_run():
    start_time=time.time()
    trainer = pl.Trainer(
        max_epochs=10,
        devices=1,
        accelerator="gpu",
        log_every_n_steps=1,
        callbacks=[
            pl.callbacks.ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1),
        ]
    )
    # Train the model
    trainer.fit(model, train_loader, val_loader)

    # Log final metrics
    val_loss = trainer.callback_metrics["val_loss"]
    val_mae = trainer.callback_metrics["val_mae"]
    val_rmse = trainer.callback_metrics["val_rmse"]

    mlflow.log_metric("val_loss", val_loss.item())
    mlflow.log_metric("val_mae", val_mae.item())
    mlflow.log_metric("val_rmse", val_rmse.item())

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Character-level model complete. Total runtime: {total_time:.2f} seconds")
print("Training complete.")




import matplotlib.pyplot as plt

# Plot Train vs Validation MAE
plt.figure(figsize=(10, 6))
plt.plot(model.train_mae_list, label="Train MAE")
plt.plot(model.val_mae_list, label="Validation MAE")
plt.xlabel("Epochs")
plt.ylabel("Mean Absolute Error (MAE)")
plt.title("Train vs Validation MAE per Epoch")
plt.legend()
plt.grid()
plt.show()

# Plot Train vs Validation RMSE
plt.figure(figsize=(10, 6))
plt.plot(model.train_rmse_list, label="Train RMSE")
plt.plot(model.val_rmse_list, label="Validation RMSE")
plt.xlabel("Epochs")
plt.ylabel("Root Mean Squared Error (RMSE)")
plt.title("Train vs Validation RMSE per Epoch")
plt.legend()
plt.grid()
plt.show()



# Testing the model
from tqdm import tqdm
# # Load the solution file with test labels
# solution_path = "/kaggle/input/home-depot-deep-learning-solutions/solution.csv"
# solution_df = pd.read_csv(solution_path)

# # Filter out rows with "Ignored" in the Usage column
# solution_df = solution_df[solution_df["Usage"] != "Ignored"]
# # Merge test_df with solution_df to get the true relevance labels
# test_df = test_df.merge(solution_df[["id", "relevance"]], on="id", how="left")
# test_df = test_df.dropna(subset=["relevance"]).reset_index(drop=True)

# Adjust Dataset class to include relevance for testing
class HomeDepotTestDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]  # Safely access the row with .iloc
        query = torch.tensor(row['search_term_chars'], dtype=torch.long)  # Search term indices
        description = torch.tensor(row['item_description_chars'], dtype=torch.long)  # Combined text indices
        relevance = torch.tensor(row['relevance'], dtype=torch.float32)  # Relevance score
        return query, description, relevance


# Testing the model
def test_model(model, test_data):
    model.eval()
    test_dataset = HomeDepotTestDataset(test_data)
    print("Create dataset completed")
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4)  # Adjust workers as per system capacity
    print("Create dataloader completed")

    all_predictions = []  # To store all predictions
    all_targets = []  # To store all true relevance scores

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing", unit="batch"):
            # Unpack the batch (adjusted to match the tuple structure)
            query, description, relevance = batch
            query = query.to(model.device)
            description = description.to(model.device)
            relevance = relevance.to(model.device)

            # Forward pass
            preds = model(query, description)

            # Accumulate predictions and targets
            all_predictions.append(preds)
            all_targets.append(relevance)

    # Concatenate predictions and targets for the entire dataset
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Calculate metrics using model's loss functions
    mse = model.MSE(all_predictions, all_targets)
    rmse = torch.sqrt(mse)
    mae = model.MAE(all_predictions, all_targets)

    return all_predictions.cpu().numpy(), rmse.item(), mae.item()



# Run testing
start_time = time.time()
test_predictions,  test_rmse, test_mae = test_model(model, test_df)
end_time = time.time()
print("Total test time:", end_time - start_time)
print(f"Test MAE: {test_mae:.4f}")
print(f"Test RMSE: {test_rmse:.4f}")

test_df["predicted_relevance"] = test_predictions

# Save predictions to a CSV file
output_test_path = os.path.join(output_dir, "test_predictions_with_loss.csv")
test_df.to_csv(output_test_path, index=False)

print(f"Test predictions saved to {output_test_path}.")




from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Extracting features for train and validation
print("Extracting features for train and validation...")

train_features = []
train_targets = []
for batch in tqdm(train_loader, desc="Processing train batches", unit="batch"):
    search_term_chars = batch[0].to(model.device)  # Assuming first element is `search_term_chars`
    item_description_chars = batch[1].to(model.device)  # Assuming second element is `item_description_chars`
    relevance = batch[2]  # Assuming third element is `relevance`

    features = model.extract_features(search_term_chars, item_description_chars)
    train_features.append(features)
    train_targets.append(relevance.cpu().numpy())

val_features = []
val_targets = []
for batch in tqdm(val_loader, desc="Processing validation batches", unit="batch"):
    search_term_chars = batch[0].to(model.device)  # Assuming first element is `search_term_chars`
    item_description_chars = batch[1].to(model.device)  # Assuming second element is `item_description_chars`
    relevance = batch[2]  # Assuming third element is `relevance`

    features = model.extract_features(search_term_chars, item_description_chars)
    val_features.append(features)
    val_targets.append(relevance.cpu().numpy())

# Concatenate features and targets
train_features = np.concatenate(train_features, axis=0)
train_targets = np.concatenate(train_targets, axis=0)
val_features = np.concatenate(val_features, axis=0)
val_targets = np.concatenate(val_targets, axis=0)





test_dataset = HomeDepotTestDataset(test_df)
print("Create dataset completed")
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4)  # Adjust workers as per system capacity
print("Create dataloader completed")
# Extracting features for test
print("Extracting features for test...")
test_features = []
test_targets = []
for batch in tqdm(test_loader, desc="Processing test batches", unit="batch"):
    search_term_chars = batch[0].to(model.device)  # Assuming first element is `search_term_chars`
    item_description_chars = batch[1].to(model.device)  # Assuming second element is `item_description_chars`
    relevance = batch[2]  # Assuming third element is `relevance`

    features = model.extract_features(search_term_chars, item_description_chars)
    test_features.append(features)
    test_targets.append(relevance.cpu().numpy())

test_features = np.concatenate(test_features, axis=0)
test_targets = np.concatenate(test_targets, axis=0)




# Train Random Forest
start_time_rf = time.time()
print("Training Random Forest...")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, max_depth=6)
rf_model.fit(train_features, train_targets)

# Predictions
rf_train_predictions = rf_model.predict(train_features)
rf_val_predictions = rf_model.predict(val_features)
rf_test_predictions = rf_model.predict(test_features)

# Metrics
rf_train_rmse = np.sqrt(mean_squared_error(train_targets, rf_train_predictions))
rf_train_mae = mean_absolute_error(train_targets, rf_train_predictions)

rf_val_rmse = np.sqrt(mean_squared_error(val_targets, rf_val_predictions))
rf_val_mae = mean_absolute_error(val_targets, rf_val_predictions)

rf_test_rmse = np.sqrt(mean_squared_error(test_targets, rf_test_predictions))
rf_test_mae = mean_absolute_error(test_targets, rf_test_predictions)

end_time_rf = time.time()

# Results
print(f"Random Forest - Train RMSE: {rf_train_rmse:.4f}, Train MAE: {rf_train_mae:.4f}")
print(f"Random Forest - Validation RMSE: {rf_val_rmse:.4f}, Validation MAE: {rf_val_mae:.4f}")
print(f"Random Forest - Test RMSE: {rf_test_rmse:.4f}, Test MAE: {rf_test_mae:.4f}")
print(f"Total Random Forest Training Time: {end_time_rf - start_time_rf:.2f} seconds")



import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import time

# Train XGBoost with updated hyperparameters
start_time_xgb = time.time()
print("Training XGBoost with reduced overfitting...")
xgb_model = XGBRegressor(
    n_estimators=100,       # Number of estimators (trees)
    max_depth=5,            # Reduced depth to limit tree complexity
    learning_rate=0.05,      # Learning rate remains the same
    min_child_weight=5,     # Increased minimum child weight
    reg_alpha=0.5,          # Added L1 regularization
    reg_lambda=1.0,         # Added L2 regularization
    random_state=42,        # For reproducibility
    n_jobs=-1               # Parallel processing
)
xgb_model.fit(train_features, train_targets, eval_set=[(val_features, val_targets)], verbose=False)

# Evaluate metrics per 20 estimators
n_estimators = range(20, xgb_model.n_estimators + 1, 20)
train_rmse_per_stage = []
train_mae_per_stage = []
val_rmse_per_stage = []
val_mae_per_stage = []

for n in n_estimators:
    # Predict using the first `n` estimators
    xgb_model.set_params(n_estimators=n)
    train_preds = xgb_model.predict(train_features, iteration_range=(0, n))
    val_preds = xgb_model.predict(val_features, iteration_range=(0, n))

    # Compute metrics
    train_rmse = np.sqrt(mean_squared_error(train_targets, train_preds))
    train_mae = mean_absolute_error(train_targets, train_preds)
    val_rmse = np.sqrt(mean_squared_error(val_targets, val_preds))
    val_mae = mean_absolute_error(val_targets, val_preds)

    train_rmse_per_stage.append(train_rmse)
    train_mae_per_stage.append(train_mae)
    val_rmse_per_stage.append(val_rmse)
    val_mae_per_stage.append(val_mae)

# Final Metrics on Full Estimator Set
final_train_preds = xgb_model.predict(train_features)
final_val_preds = xgb_model.predict(val_features)
final_test_preds = xgb_model.predict(test_features)

final_train_rmse = np.sqrt(mean_squared_error(train_targets, final_train_preds))
final_train_mae = mean_absolute_error(train_targets, final_train_preds)

final_val_rmse = np.sqrt(mean_squared_error(val_targets, final_val_preds))
final_val_mae = mean_absolute_error(val_targets, final_val_preds)

final_test_rmse = np.sqrt(mean_squared_error(test_targets, final_test_preds))
final_test_mae = mean_absolute_error(test_targets, final_test_preds)

end_time_xgb = time.time()

# Print Final Metrics
print("\n=== Final Metrics with Reduced Overfitting ===")
print(f"XGBoost - Train RMSE: {final_train_rmse:.4f}, Train MAE: {final_train_mae:.4f}")
print(f"XGBoost - Validation RMSE: {final_val_rmse:.4f}, Validation MAE: {final_val_mae:.4f}")
print(f"XGBoost - Test RMSE: {final_test_rmse:.4f}, Test MAE: {final_test_mae:.4f}")
print(f"Total XGBoost Training Time: {end_time_xgb - start_time_xgb:.2f} seconds")

# Plotting metrics per 20 estimators
plt.figure(figsize=(12, 6))

# RMSE Plot
plt.subplot(1, 2, 1)
plt.plot(n_estimators, train_rmse_per_stage, label="Train RMSE", marker='o')
plt.plot(n_estimators, val_rmse_per_stage, label="Validation RMSE", marker='o')
plt.xlabel("Number of Estimators")
plt.ylabel("RMSE")
plt.title("RMSE vs Number of Estimators")
plt.legend()
plt.grid(True)

# MAE Plot
plt.subplot(1, 2, 2)
plt.plot(n_estimators, train_mae_per_stage, label="Train MAE", marker='o')
plt.plot(n_estimators, val_mae_per_stage, label="Validation MAE", marker='o')
plt.xlabel("Number of Estimators")
plt.ylabel("MAE")
plt.title("MAE vs Number of Estimators")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()



pip install --upgrade smart_open gensim


import pandas as pd
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
from gensim.models import Word2Vec
import json
import nltk

# # Download NLTK resources if not already available
# nltk.download('punkt')
# nltk.download('stopwords')

# Define paths
output_dir = "/kaggle/working/home-depot-product-search-relevance"
train_path = os.path.join(output_dir, "train.csv")
test_path = os.path.join(output_dir, "test.csv")
descriptions_path = os.path.join(output_dir, "product_descriptions.csv")
attributes_path = os.path.join(output_dir, "attributes.csv")

# Load data
train_df = pd.read_csv(train_path, encoding="latin1", on_bad_lines="skip", engine="python")
test_df = pd.read_csv(test_path, encoding="latin1", on_bad_lines="skip", engine="python")
descriptions_df = pd.read_csv(descriptions_path)
attributes_df = pd.read_csv(attributes_path)

# Function: Clean text
def clean_text(text):
    """Cleans text by removing punctuation, special characters, and converting to lowercase."""
    text = text.lower()
    text = re.sub(r'[^\w\s#°¾]+', '', text)  # Retain #, °, and ¾
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    return text

# Add descriptions to train and test datasets
train_df = pd.merge(train_df, descriptions_df, on="product_uid", how="left")
test_df = pd.merge(test_df, descriptions_df, on="product_uid", how="left")

# Clean descriptions
train_df["product_description"] = train_df["product_description"].fillna("").apply(clean_text)
test_df["product_description"] = test_df["product_description"].fillna("").apply(clean_text)

# Add attributes to train and test datasets
def aggregate_attributes(attributes_df):
    """Aggregates attribute values for each product_uid into a single string."""
    attributes_df["value"] = attributes_df["value"].fillna("").apply(clean_text)
    attributes_aggregated = attributes_df.groupby("product_uid")["value"].apply(lambda x: " ".join(x)).reset_index()
    attributes_aggregated.rename(columns={"value": "attributes"}, inplace=True)
    return attributes_aggregated

attributes_aggregated = aggregate_attributes(attributes_df)
train_df = pd.merge(train_df, attributes_aggregated, on="product_uid", how="left")
test_df = pd.merge(test_df, attributes_aggregated, on="product_uid", how="left")
train_df["attributes"] = train_df["attributes"].fillna("")
test_df["attributes"] = test_df["attributes"].fillna("")


# Limit text to the first N words
def limit_words(text, max_words=50):
    """Limits the input text to the first max_words words."""
    words = text.split()
    return ' '.join(words[:max_words])

# Update the combine_text_without_search_term function
def combine_text_without_search_term(row):
    """Combines product_title, limited product_description, and limited attributes."""
    product_description = limit_words(row['product_description'], max_words=70)
    attributes = limit_words(row['attributes'], max_words=70)
    text = f"{row['product_title']} {product_description} {attributes}"
    return text

# Function: Tokenize text into words/character combinations
def tokenize_text_with_combinations(text):
    """
    Tokenizes text into words, preserving character combinations like #, °, ¾, etc.
    """
    tokens = word_tokenize(text)
    # Filter tokens to include alphanumeric words and special combinations
    tokens = [token for token in tokens if re.match(r'[\w#°¾]+', token)]
    return tokens

# Tokenize search_term separately for words/character-combinations
train_df["search_term_tokens"] = train_df["search_term"].apply(
    lambda x: tokenize_text_with_combinations(clean_text(x))
)
test_df["search_term_tokens"] = test_df["search_term"].apply(
    lambda x: tokenize_text_with_combinations(clean_text(x))
)

# Combine product title, description, and attributes for tokenization
train_df["combined_tokens"] = train_df.apply(
    lambda row: tokenize_text_with_combinations(
        clean_text(combine_text_without_search_term(row))
    ),
    axis=1,
)
test_df["combined_tokens"] = test_df.apply(
    lambda row: tokenize_text_with_combinations(
        clean_text(combine_text_without_search_term(row))
    ),
    axis=1,
)

# Save processed tokenized data
train_df.to_csv(f"{output_dir}/train_tokenized_with_combinations.csv", index=False)
test_df.to_csv(f"{output_dir}/test_tokenized_with_combinations.csv", index=False)

print("Step A complete: Data tokenized with word/character combinations and saved.")






from gensim.models import Word2Vec

# Prepare sentences for training Word2Vec
all_sentences = train_df["search_term_tokens"].tolist() + \
                train_df["combined_tokens"].tolist() + \
                test_df["search_term_tokens"].tolist() + \
                test_df["combined_tokens"].tolist()

# Train Word2Vec embeddings
word2vec_model = Word2Vec(
    sentences=all_sentences,
    vector_size=300,  # Dimension of the embeddings
    window=5,         # Context window size
    min_count=2,      # Minimum frequency of words to include
    workers=4         # Number of worker threads
)

# Save Word2Vec model and embeddings
word2vec_model.save(f"{output_dir}/word2vec_embeddings_with_combinations.model")
print("Step B complete: Word2Vec embeddings trained and saved.")



# Build vocabulary
def build_vocab(tokens_list):
    """Builds a vocabulary from tokenized text."""
    all_tokens = [token for tokens in tokens_list for token in tokens]
    token_counts = Counter(all_tokens)
    sorted_tokens = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)
    vocab = {token: idx + 1 for idx, (token, _) in enumerate(sorted_tokens)}  # Reserve 0 for padding
    return vocab


vocab = build_vocab(all_sentences)
# Create an embedding matrix aligned with the vocabulary
embedding_dim = word2vec_model.vector_size
vocab_size = len(vocab) + 1  # Reserve 0 for padding
embedding_matrix = np.zeros((vocab_size, embedding_dim))  # Initialize with zeros

# Fill the embedding matrix with Word2Vec vectors
for word, idx in vocab.items():
    if word in word2vec_model.wv:
        embedding_matrix[idx] = word2vec_model.wv[word]  # Assign Word2Vec vector
    else:
        embedding_matrix[idx] = np.random.normal(size=(embedding_dim,))  # Random vector for unknown words

# Validate and filter the vocabulary
assert max(vocab.values()) < embedding_matrix.shape[0], \
    "Vocabulary indices exceed embedding matrix size."

# Filter tokens
valid_vocab_tokens = {token: idx for token, idx in vocab.items() if idx < embedding_matrix.shape[0]}

# Convert tokens to indices
def tokens_to_indices(tokens, vocab):
    """Convert tokens to indices based on the filtered vocabulary."""
    return [vocab.get(token, 0) for token in tokens]  # Default to 0 if token not in vocab

# Apply mapping
train_df["search_term_indices"] = train_df["search_term_tokens"].apply(lambda x: tokens_to_indices(x, valid_vocab_tokens))
train_df["combined_indices"] = train_df["combined_tokens"].apply(lambda x: tokens_to_indices(x, valid_vocab_tokens))
test_df["search_term_indices"] = test_df["search_term_tokens"].apply(lambda x: tokens_to_indices(x, valid_vocab_tokens))
test_df["combined_indices"] = test_df["combined_tokens"].apply(lambda x: tokens_to_indices(x, valid_vocab_tokens))

# Validation
print("Token-to-index mapping completed.")
print(f"Example search_term_tokens: {train_df.iloc[0]['search_term_tokens']}")
print(f"Example search_term_indices: {train_df.iloc[0]['search_term_indices']}")



train_df


import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec
import numpy as np
import pandas as pd
from torch.nn.utils.rnn import pad_sequence




# Define Dataset
class HomeDepotDataset(Dataset):
    def __init__(self, data):
        """
        Initialize the dataset.
        Args:
            data (pd.DataFrame): The dataframe containing the dataset.
        """
        self.data = data

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieves a single data sample by index.
        Args:
            idx (int): The index of the data sample.
        Returns:
            dict: A dictionary containing input and target tensors.
        """
        row = self.data.iloc[idx]
        search_indices = torch.tensor(row["search_term_indices"], dtype=torch.long)
        combined_indices = torch.tensor(row["combined_indices"], dtype=torch.long)
        relevance = torch.tensor(row["relevance"], dtype=torch.float32)

        return {
            "search_term": search_indices,
            "combined_text": combined_indices,
            "relevance": relevance
        }

    @staticmethod
    def pad_or_truncate(sequence, max_len):
        """Truncate or pad sequence to max_len."""
        if len(sequence) > max_len:
            return sequence[:max_len]
        else:
            return sequence + [0] * (max_len - len(sequence))

# Collate function to handle NoneType or invalid data
def collate_fn(batch):
    search_terms = [sample["search_term"] for sample in batch]
    combined_texts = [sample["combined_text"] for sample in batch]
    relevances = [sample["relevance"] for sample in batch]

    # Pad sequences to a minimum length of 20
    min_length = 20
    search_terms_padded = pad_sequence(
        search_terms, batch_first=True, padding_value=0
    )
    combined_texts_padded = pad_sequence(
        combined_texts, batch_first=True, padding_value=0
    )

    # Ensure minimum length
    search_terms_padded = F.pad(
        search_terms_padded, (0, max(0, min_length - search_terms_padded.size(1))), value=0
    )
    combined_texts_padded = F.pad(
        combined_texts_padded, (0, max(0, min_length - combined_texts_padded.size(1))), value=0
    )

    relevances_tensor = torch.stack(relevances)

    return {
        "search_term": search_terms_padded,
        "combined_text": combined_texts_padded,
        "relevance": relevances_tensor,
    }

train_df["search_term_indices"] = train_df["search_term_indices"].apply(lambda x: x if len(x) > 0 else [0])
train_df["combined_indices"] = train_df["combined_indices"].apply(lambda x: x if len(x) > 0 else [0])

# Similarly for test data
test_df["search_term_indices"] = test_df["search_term_indices"].apply(lambda x: x if len(x) > 0 else [0])
test_df["combined_indices"] = test_df["combined_indices"].apply(lambda x: x if len(x) > 0 else [0])

train_data, val_data = train_test_split(train_df, test_size=0.2, random_state=42)
# Create DataLoader
train_dataset = HomeDepotDataset(train_data)
val_dataset = HomeDepotDataset(val_data)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)





import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import mlflow

# Define Siamese LSTM Network
class SiameseLSTM(pl.LightningModule):
    def __init__(self, embedding_matrix, hidden_size=128, num_layers=2, dropout=0.3, learning_rate=1e-3):
        super(SiameseLSTM, self).__init__()
        
        vocab_size, embedding_dim = embedding_matrix.shape
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.embedding.weight.data.copy_(torch.tensor(embedding_matrix))
        self.embedding.weight.requires_grad = False  # Freeze embeddings
        
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=True,
            batch_first=True
        )
        
        self.fc = nn.Linear(hidden_size * 2, 1)
        self.criterion = nn.MSELoss()
        self.learning_rate = learning_rate

    def forward_one_branch(self, x):
        x = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        _, (hidden, _) = self.lstm(x)
        x = torch.cat((hidden[-2], hidden[-1]), dim=-1)  # Concatenate last forward & backward hidden states
        return x

    def forward(self, search_term, combined_text):
        encoded_search = self.forward_one_branch(search_term)
        encoded_combined = self.forward_one_branch(combined_text)
        distance = torch.abs(encoded_search - encoded_combined)
        return self.fc(distance).squeeze()

    def training_step(self, batch, batch_idx):
        search_term, combined_text, relevance = batch["search_term"], batch["combined_text"], batch["relevance"]
        predictions = self(search_term, combined_text)
        loss = self.criterion(predictions, relevance)
        mae = F.l1_loss(predictions, relevance)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        self.log("train_mae", mae, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        search_term, combined_text, relevance = batch["search_term"], batch["combined_text"], batch["relevance"]
        predictions = self(search_term, combined_text)
        loss = self.criterion(predictions, relevance)
        mae = F.l1_loss(predictions, relevance)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_mae", mae, on_epoch=True, prog_bar=True)
        return loss


    def on_train_epoch_end(self):
        train_loss = self.trainer.callback_metrics["train_loss"].item()
        train_mae = self.trainer.callback_metrics["train_mae"].item()
        train_rmse = train_loss ** 0.5
        print(f"Epoch {self.current_epoch + 1}: Train MSE: {train_loss:.4f}, Train MAE: {train_mae:.4f}, Train RMSE: {train_rmse:.4f}")
        mlflow.log_metric("train_loss", train_loss, step=self.current_epoch + 1)
        mlflow.log_metric("train_mae", train_mae, step=self.current_epoch + 1)
        mlflow.log_metric("train_rmse", train_rmse, step=self.current_epoch + 1)

    def on_validation_epoch_end(self):
        val_loss = self.trainer.callback_metrics["val_loss"].item()
        val_mae = self.trainer.callback_metrics["val_mae"].item()
        val_rmse = val_loss ** 0.5
        print(f"Epoch {self.current_epoch + 1}: Validation MSE: {val_loss:.4f}, Validation MAE: {val_mae:.4f}, Validation RMSE: {val_rmse:.4f}")
        mlflow.log_metric("val_loss", val_loss, step=self.current_epoch + 1)
        mlflow.log_metric("val_mae", val_mae, step=self.current_epoch + 1)
        mlflow.log_metric("val_rmse", val_rmse, step=self.current_epoch + 1)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=5e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=3, factor=0.5, verbose=True
    )
        return {
            'optimizer' : optimizer,
            'lr_scheduler' : {
                'scheduler' : scheduler,
                'monitor' : 'val_loss',
                'frequency' : 1
            }
        }

# Train the model
model_sn = SiameseLSTM(embedding_matrix=embedding_matrix)

mlflow.set_experiment("Home Depot Relevance Training-Words")

with mlflow.start_run() as run:
    trainer = pl.Trainer(
        max_epochs=15,
        devices=1,
        accelerator="gpu",
        log_every_n_steps=1,
        callbacks=[
            pl.callbacks.ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1),
        ]
    )

    trainer.fit(model_sn, train_loader, val_loader)

print("Training complete.")


import matplotlib.pyplot as plt
from mlflow import MlflowClient

run_id = mlflow.last_active_run().info.run_id
client = MlflowClient()
metrics = ["train_mae","train_rmse","val_mae","val_rmse"]
met_hist = {}
for met in metrics:
    hist = client.get_metric_history(run_id, met)
    met_hist[met] = {x.step : x.value for x in hist}

# Extract epochs and metrics
epochs = list(met_hist['train_mae'].keys())
train_mae = list(met_hist['train_mae'].values())
val_mae = list(met_hist['val_mae'].values())
train_rmse = list(met_hist['train_rmse'].values())
val_rmse = list(met_hist['val_rmse'].values())

# Plot MAE
plt.figure(figsize=(10, 5))
plt.plot(epochs, train_mae, label="Train MAE", marker='o')
plt.plot(epochs, val_mae, label="Validation MAE", marker='o')
plt.xlabel("Epochs")
plt.ylabel("Mean Absolute Error (MAE)")
plt.title("Train and Validation MAE")
plt.legend()
plt.grid(True)
plt.show()

# Plot RMSE
plt.figure(figsize=(10, 5))
plt.plot(epochs, train_rmse, label="Train RMSE", marker='o')
plt.plot(epochs, val_rmse, label="Validation RMSE", marker='o')
plt.xlabel("Epochs")
plt.ylabel("Root Mean Squared Error (RMSE)")
plt.title("Train and Validation RMSE")
plt.legend()
plt.grid(True)
plt.show()


solution_path = "/kaggle/input/home-depot-deep-learning-solutions/solution.csv"
solution_df = pd.read_csv(solution_path)

# Filter out rows with "Ignored" in the Usage column
solution_df = solution_df[solution_df["Usage"] != "Ignored"]
# Merge test_df with solution_df to get the true relevance labels
test_df = test_df.merge(solution_df[["id", "relevance"]], on="id", how="left")
# test_df = test_df.drop(columns=["relevance_y"])

print(test_df.head(5))
print(solution_df.head(5))
# test_df = test_df.rename(columns={"relevance_x": "relevance"})

test_df = test_df.dropna(subset=["relevance"]).reset_index(drop=True)

# test_dataset = HomeDepotDataset(test_df)

# test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)

# Testing the model
def test_model(model, test_data):
    model.eval()
    test_dataset = HomeDepotDataset(test_data)
    print("create dataset completed")
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4, collate_fn=collate_fn)
    print("create dataloader completed")
    predictions = []
    total_loss = 0
    total_mae =0
    criterion = nn.MSELoss()  # Define the same loss function used during training
    num_batches = 0

    with torch.no_grad():
        for batch in test_loader:
            preds = model(batch["search_term"], batch["combined_text"])
            predictions.extend(preds.cpu().numpy())

            # Calculate loss for the current batch
            if "relevance" in batch:
                loss = criterion(preds, batch["relevance"].to(model.device))
                mae = torch.mean(torch.abs(preds - batch["relevance"]))
                total_loss += loss.item()
                num_batches += 1
                total_mae+=mae

    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    avg_mae = total_mae / num_batches if num_batches > 0 else 0
    print("Average MSE:", avg_loss)
    avg_rmse = (avg_loss ** 0.5)
    print("Average MAE:", avg_mae)
    print("Average RMSE:", avg_rmse)

    return predictions, avg_loss, avg_mae, avg_rmse


# Run testing
test_predictions, test_loss, test_mae, test_rmse = test_model(model_sn, test_df)
print(f"Test Loss: {test_loss:.4f}")
test_df["predicted_relevance"] = test_predictions

# Save predictions to a CSV file
output_test_path = os.path.join(output_dir, "test_predictions_with_loss_word.csv")
test_df.to_csv(output_test_path, index=False)

print(f"Test predictions saved to {output_test_path}.")



test_df


!pip install sentence-transformers


def create_combined_text(df):
    """
    Combine product_title, product_description, and attributes into a single column.
    Args:
        df (pd.DataFrame): The DataFrame containing the text columns.
    Returns:
        pd.DataFrame: DataFrame with a new 'combined_text' column.
    """
    df["combined_text"] = (
        df["product_title"].fillna("") + " " +
        df["product_description"].fillna("") + " " +
        df["attributes"].fillna("")
    )
    return df

# Apply the function to create combined_text for all datasets
train_df = create_combined_text(train_df)
test_df = create_combined_text(test_df)


from sentence_transformers import SentenceTransformer
import pandas as pd
import torch

# Load the paraphrase-mpnet-base-v2 model
sbert_model = SentenceTransformer('paraphrase-MiniLM-L12-v2')

def encode_with_sbert(df, text_columns):
    """
    # Encode text columns in a DataFrame using SBERT.
    Args:
        df (pd.DataFrame): The DataFrame containing text data.
        text_columns (list): List of column names to encode.
    Returns:
        pd.DataFrame: DataFrame with additional columns for embeddings.
    """
    for col in text_columns:
        print(f"Encoding column: {col} using paraphrase-MiniLM-L12-v2...")
        embeddings = sbert_model.encode(df[col].tolist(), show_progress_bar=True)
        df[f"{col}_embedding"] = list(embeddings)
    return df

# Encode search_term and combined_text
text_columns = ["search_term", "combined_text"]
train_df = encode_with_sbert(train_df, text_columns)
test_df = encode_with_sbert(test_df, text_columns)



class SBERTDataset(Dataset):
    def __init__(self, data, is_test=False):
        """
        Initialize the dataset with precomputed SBERT embeddings.
        Args:
            data (pd.DataFrame): The DataFrame containing the dataset.
            is_test (bool): Whether this is a test dataset.
        """
        self.data = data
        self.is_test = is_test

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        search_embedding = torch.tensor(row["search_term_embedding"], dtype=torch.float)
        combined_embedding = torch.tensor(row["combined_text_embedding"], dtype=torch.float)

        sample = {
            "search_embedding": search_embedding,
            "combined_embedding": combined_embedding,
        }

        if not self.is_test:
            sample["relevance"] = torch.tensor(row["relevance"], dtype=torch.float)

        return sample



import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import mlflow

class SBERTSiameseNetworkLSTM(pl.LightningModule):
    def __init__(self, embedding_dim=384, hidden_size=128, num_layers=2, dropout=0.3, learning_rate=1e-3):
        """
        Siamese Network for SBERT embeddings using LSTM layers.
        Args:
            embedding_dim (int): Dimensionality of SBERT embeddings (384 for MiniLM).
            hidden_size (int): Hidden size of LSTM layers.
            num_layers (int): Number of LSTM layers.
            dropout (float): Dropout rate.
            learning_rate (float): Learning rate for the optimizer.
        """
        super(SBERTSiameseNetworkLSTM, self).__init__()
        self.embedding_dim = embedding_dim
        
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=True,
            batch_first=True
        )
        
        self.fc1 = nn.Linear(hidden_size * 2, 128)  # Bidirectional LSTM output size
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 1)
        
        self.criterion = nn.MSELoss()
        self.learning_rate = learning_rate
    
    def forward_one_branch(self, embeddings):
        """
        Forward pass for one branch of the Siamese network.
        Args:
            embeddings (torch.Tensor): Input embeddings (batch_size, embedding_dim).
        Returns:
            torch.Tensor: Encoded representation of the input.
        """
        _, (hidden, _) = self.lstm(embeddings)
        x = torch.cat((hidden[-2], hidden[-1]), dim=-1)  # Concatenate last forward & backward hidden states
        return x
    
    def forward(self, search_embedding, combined_embedding):
        encoded_search = self.forward_one_branch(search_embedding)
        encoded_combined = self.forward_one_branch(combined_embedding)
        distance = torch.abs(encoded_search - encoded_combined)
        x = F.relu(self.fc1(distance))
        x = self.dropout(x)
        relevance_score = self.fc2(x)
        return relevance_score.squeeze()

    def training_step(self, batch, batch_idx):
        search_embedding = batch["search_embedding"]
        combined_embedding = batch["combined_embedding"]
        relevance = batch["relevance"]
        predictions = self.forward(search_embedding, combined_embedding)
        loss = self.criterion(predictions, relevance)
        mae = torch.mean(torch.abs(predictions - relevance))
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        self.log("train_mae", mae, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        search_embedding = batch["search_embedding"]
        combined_embedding = batch["combined_embedding"]
        relevance = batch["relevance"]
        predictions = self.forward(search_embedding, combined_embedding)
        loss = self.criterion(predictions, relevance)
        mae = torch.mean(torch.abs(predictions - relevance))
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_mae", mae, on_epoch=True, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        """
        Log training metrics to MLflow.
        """
        train_loss = self.trainer.callback_metrics["train_loss"].item()
        train_mae = self.trainer.callback_metrics["train_mae"].item()
        train_rmse = train_loss ** 0.5

        print(f"Epoch {self.current_epoch + 1}: Train MSE: {train_loss:.4f}, "
              f"Train MAE: {train_mae:.4f}, Train RMSE: {train_rmse:.4f}")

        mlflow.log_metric("train_loss", train_loss, step=self.current_epoch + 1)
        mlflow.log_metric("train_mae", train_mae, step=self.current_epoch + 1)
        mlflow.log_metric("train_rmse", train_rmse, step=self.current_epoch + 1)

    def on_validation_epoch_end(self):
        """
        Log validation metrics to MLflow.
        """
        val_loss = self.trainer.callback_metrics["val_loss"].item()
        val_mae = self.trainer.callback_metrics["val_mae"].item()
        val_rmse = val_loss ** 0.5

        print(f"Epoch {self.current_epoch + 1}: Validation MSE: {val_loss:.4f}, "
              f"Validation MAE: {val_mae:.4f}, Validation RMSE: {val_rmse:.4f}")

        mlflow.log_metric("val_loss", val_loss, step=self.current_epoch + 1)
        mlflow.log_metric("val_mae", val_mae, step=self.current_epoch + 1)
        mlflow.log_metric("val_rmse", val_rmse, step=self.current_epoch + 1)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=5e-4, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
            }
        }



from torch.utils.data import DataLoader

train, val = train_test_split(train_df, test_size=0.15, random_state=42)

# Create datasets and dataloaders
train_dataset = SBERTDataset(train)
val_dataset = SBERTDataset(val)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=3)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=3)

# Initialize and train the model


mlflow.set_experiment("Home Depot Relevance Training-Words SBERT")

with mlflow.start_run() as run:
    model_sbert = SBERTSiameseNetworkLSTM()
    trainer = pl.Trainer(
        max_epochs=10,
        devices=1,
        accelerator="gpu",
        log_every_n_steps=1,
        callbacks=[
            pl.callbacks.ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1),
        ]
    )

    trainer.fit(model_sbert, train_loader, val_loader)

print("Training complete.")



mlflow.last_active_run()


import matplotlib.pyplot as plt
from mlflow import MlflowClient

run_id = mlflow.last_active_run().info.run_id
client = MlflowClient()
metrics = ["train_mae","train_rmse","val_mae","val_rmse"]
met_hist = {}
for met in metrics:
    hist = client.get_metric_history(run_id, met)
    met_hist[met] = {x.step : x.value for x in hist}

# Extract epochs and metrics
epochs = list(met_hist['train_mae'].keys())
train_mae = list(met_hist['train_mae'].values())
val_mae = list(met_hist['val_mae'].values())
train_rmse = list(met_hist['train_rmse'].values())
val_rmse = list(met_hist['val_rmse'].values())

# Plot MAE
plt.figure(figsize=(10, 5))
plt.plot(epochs, train_mae, label="Train MAE", marker='o')
plt.plot(epochs, val_mae, label="Validation MAE", marker='o')
plt.xlabel("Epochs")
plt.ylabel("Mean Absolute Error (MAE)")
plt.title("Train and Validation MAE")
plt.legend()
plt.grid(True)
plt.show()

# Plot RMSE
plt.figure(figsize=(10, 5))
plt.plot(epochs, train_rmse, label="Train RMSE", marker='o')
plt.plot(epochs, val_rmse, label="Validation RMSE", marker='o')
plt.xlabel("Epochs")
plt.ylabel("Root Mean Squared Error (RMSE)")
plt.title("Train and Validation RMSE")
plt.legend()
plt.grid(True)
plt.show()


from tqdm.notebook import tqdm


# Create test DataLoader
test_dataset = SBERTDataset(test_df, is_test=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Make predictions
model_sbert.eval()
predictions = []

with torch.no_grad():
    for batch in tqdm(test_loader):
        search_embedding = batch["search_embedding"]
        combined_embedding = batch["combined_embedding"]
        preds = model_sbert(search_embedding, combined_embedding).squeeze(-1)
        predictions.extend(preds.cpu().numpy())

# Add predictions to test_df
test_df["predicted_relevance"] = predictions

# Save predictions
output_path = "test_predictions_with_mpnet.csv"
test_df.to_csv(output_path, index=False)
print(f"Predictions saved to {output_path}.")



test_df = test_df.loc[:, ~test_df.columns.duplicated()]
test_df


# Testing the SBERT model
def test_sbert_model(model, test_data):
    """
    Test the SBERT model on the test dataset.
    Args:
        model: Trained SBERT model.
        test_data (pd.DataFrame): Test DataFrame with precomputed SBERT embeddings.
    Returns:
        tuple: Predictions, average loss, average MAE, and RMSE.
    """
    model_sbert.eval()
    test_dataset = SBERTDataset(test_data, is_test=False)
    print("Dataset creation completed.")

    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)
    print("Dataloader creation completed.")

    predictions = []
    total_loss = 0
    total_mae = 0
    criterion = nn.MSELoss()  # Define the same loss function used during training
    num_batches = 0

    with torch.no_grad():
        for batch in test_loader:
            search_embedding = batch["search_embedding"].to(model_sbert.device)
            combined_embedding = batch["combined_embedding"].to(model_sbert.device)

            # Get predictions from the model
            preds = model_sbert(search_embedding, combined_embedding).squeeze(-1)
            predictions.extend(preds.cpu().numpy())

            # Calculate loss and MAE if relevance is available
            if "relevance" in batch:
                relevance = batch["relevance"].to(model.device)
                loss = criterion(preds, relevance)
                mae = torch.mean(torch.abs(preds - relevance))
                total_loss += loss.item()
                total_mae += mae.item()
                num_batches += 1

    # Compute average metrics
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    avg_mae = total_mae / num_batches if num_batches > 0 else 0
    avg_rmse = avg_loss ** 0.5 if avg_loss > 0 else 0

    print("Average MSE:", avg_loss)
    print("Average MAE:", avg_mae)
    print("Average RMSE:", avg_rmse)

    return predictions, avg_loss, avg_mae, avg_rmse


# Run testing
test_predictions, test_loss, test_mae, test_rmse = test_sbert_model(model, test_df)
print(f"Test Loss: {test_loss:.4f}")
# test_df["predicted_relevance"] = test_predictions

# # Save predictions to a CSV file
# output_test_path = "test_predictions_with_sbert.csv"
# test_df.to_csv(output_test_path, index=False)z

# print(f"Test predictions saved to {output_test_path}.")



from tqdm.notebook import tqdm


def extract_features(feature_extractor, data_loader):
    """
    Extract features using the Siamese network.
    Args:
        feature_extractor: Pre-trained SiameseNetwork.
        data_loader: DataLoader for the dataset.
    Returns:
        tuple: Extracted features and corresponding labels.
    """
    feature_extractor.eval()
    features = []
    labels = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="processing batches", unit="batch"):
            search_term = batch["search_term"].to(feature_extractor.device)
            combined_text = batch["combined_text"].to(feature_extractor.device)
            relevance = batch["relevance"]

            # Extract features
            encoded_search = feature_extractor.forward_one_branch(search_term)
            encoded_combined = feature_extractor.forward_one_branch(combined_text)

            # Concatenate features
            concatenated_features = torch.cat((encoded_search, encoded_combined), dim=1).cpu().numpy()
            features.append(concatenated_features)
            labels.extend(relevance.cpu().numpy())

    features = np.vstack(features)
    labels = np.array(labels)
    return features, labels



# Create datasets and dataloaders
train_dataset = HomeDepotDataset(train_data)
val_dataset = HomeDepotDataset(val_data)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)
# Extract features for training and validation sets
train_features, train_labels = extract_features(model_sn, train_loader)
val_features, val_labels = extract_features(model_sn, val_loader)



test_dataset = HomeDepotDataset(test_df)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4, collate_fn=collate_fn)
test_features, test_labels = extract_features(model_sn, test_loader)


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Initialize the Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=3, random_state=42)

# Train the model
rf_model.fit(train_features, train_labels)

# Make predictions
val_predictions = rf_model.predict(val_features)

# Evaluate the model
mse = mean_squared_error(val_labels, val_predictions)
mae = mean_absolute_error(val_labels, val_predictions)
rmse = mse ** 0.5

print(f"RF Validation MSE: {mse:.4f}")
print(f"RF Validation MAE: {mae:.4f}")
print(f"RF Validation RMSE: {rmse:.4f}")



# Make predictions
train_predictions = rf_model.predict(train_features)

# Evaluate the model
mse = mean_squared_error(train_labels, train_predictions)
mae = mean_absolute_error(train_labels, train_predictions)
rmse = mse ** 0.5

print(f"RF Train MSE: {mse:.4f}")
print(f"RF Train MAE: {mae:.4f}")
print(f"RF Train RMSE: {rmse:.4f}")


test_predictions = rf_model.predict(test_features)

# Evaluate the model
mse = mean_squared_error(test_labels, test_predictions)
mae = mean_absolute_error(test_labels, test_predictions)
rmse = mse ** 0.5

print(f"RF Test MSE: {mse:.4f}")
print(f"RF Test MAE: {mae:.4f}")
print(f"RF Test RMSE: {rmse:.4f}")


from xgboost import XGBRegressor

# Initialize XGBoost Regressor
xgb_model = XGBRegressor(n_estimators=3, learning_rate=0.01, random_state=42)

# Train the model
xgb_model.fit(train_features, train_labels)

# Make predictions
val_predictions = xgb_model.predict(val_features)

# Evaluate
mse = mean_squared_error(val_labels, val_predictions)
mae = mean_absolute_error(val_labels, val_predictions)
rmse = mse ** 0.5

print(f"XgBoost Validation MSE: {mse:.4f}")
print(f"XgBoost Validation MAE: {mae:.4f}")
print(f"XgBoost Validation RMSE: {rmse:.4f}")



# Make predictions
train_predictions = xgb_model.predict(train_features)

# Evaluate the model
mse = mean_squared_error(train_labels, train_predictions)
mae = mean_absolute_error(train_labels, train_predictions)
rmse = mse ** 0.5

print(f"XgBoost Train MSE: {mse:.4f}")
print(f"XgBoost Train MAE: {mae:.4f}")
print(f"XgBoost Train RMSE: {rmse:.4f}")


test_predictions = xgb_model.predict(test_features)

# Evaluate the model
mse = mean_squared_error(test_labels, test_predictions)
mae = mean_absolute_error(test_labels, test_predictions)
rmse = mse ** 0.5

print(f"XgBoost Test MSE: {mse:.4f}")
print(f"XgBoost Test MAE: {mae:.4f}")
print(f"XgBoost Test RMSE: {rmse:.4f}")


# convert notebook to html:
!jupyter nbconvert --to html DLW3_213138787_213479686.ipynb

