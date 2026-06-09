import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


train_df = pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")
extra_df = pd.read_csv("/kaggle/input/podcast-listening-time-prediction-dataset/podcast_dataset.csv")

train_df = pd.concat([train_df, extra_df], axis=0)

test_df = pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")


# Streamline the data preprocessing function
def preprocess_data(df, is_train=True):
    data = df.copy()

    # ---- 1. Handle all missing values first ----

    # Create missing value indicators (before filling to preserve info)
    for col in [
        "Episode_Length_minutes",
        "Guest_Popularity_percentage",
    ]:
        if col in data.columns:
            data[f"{col}_missing"] = data[col].isna().astype(int)

    # Fill Episode_Length_minutes missing values (must be first)
    if "Episode_Length_minutes" in data.columns:
        median_length = data["Episode_Length_minutes"].median()
        data["Episode_Length_minutes"] = data[
            "Episode_Length_minutes"
        ].fillna(median_length)

    # Fill Guest_Popularity_percentage missing values
    if "Guest_Popularity_percentage" in data.columns:
        median_guest = data["Guest_Popularity_percentage"].median()
        data["Guest_Popularity_percentage"] = data[
            "Guest_Popularity_percentage"
        ].fillna(median_guest)

    # Fill Number_of_Ads missing values
    if "Number_of_Ads" in data.columns:
        data["Number_of_Ads"] = data["Number_of_Ads"].fillna(0)

    # ---- 2. Feature creation (after ensuring no missing values) ----

    # Episode_Length grouping
    if "Episode_Length_minutes" in data.columns:
        # Mark outliers
        data["episode_length_outlier"] = (
            data["Episode_Length_minutes"] > 180
        ).astype(int)

        # Create grouped categories (all non-null)
        data["episode_length_group"] = pd.cut(
            data["Episode_Length_minutes"],
            bins=[0, 30, 60, 120, float("inf")],
            labels=["short", "medium", "long", "extreme"],
        )
        # Convert to string for CatBoost compatibility
        data["episode_length_group"] = data[
            "episode_length_group"
        ].astype(str)

    # Number_of_Ads transformations
    if "Number_of_Ads" in data.columns:
        data["ads_log"] = np.log1p(data["Number_of_Ads"])
        data["ads_outlier"] = (data["Number_of_Ads"] > 20).astype(int)

        # Compute ad density
        if "Episode_Length_minutes" in data.columns:
            data["ads_density"] = data["Number_of_Ads"] / data[
                "Episode_Length_minutes"
            ].clip(lower=1.0)

    # Text features
    if "Episode_Title" in data.columns:
        data["title_length"] = data["Episode_Title"].str.len()
        data["title_word_count"] = (
            data["Episode_Title"].str.split().str.len()
        )

        # Keyword features
        top_keywords = [
            "interview",
            "special",
            "exclusive",
            "series",
            "live",
        ]
        data["title_has_key_term"] = (
            data["Episode_Title"]
            .str.lower()
            .apply(lambda x: any(kw in str(x) for kw in top_keywords))
            .astype(int)
        )

        data["title_has_question"] = (
            data["Episode_Title"].str.contains(r"\\?").astype(int)
        )

    # Interaction between length and popularity
    if (
        "Episode_Length_minutes" in data.columns
        and "Host_Popularity_percentage" in data.columns
    ):
        data["high_value_content"] = (
            (data["Episode_Length_minutes"] > 45)
            & (
                data["Host_Popularity_percentage"]
                > data["Host_Popularity_percentage"].median()
            )
        ).astype(int)

    # Temporal features
    if "Publication_Day" in data.columns:
        data["is_weekend"] = (
            data["Publication_Day"]
            .isin(["Saturday", "Sunday"])
            .astype(int)
        )

    if "Publication_Time" in data.columns:
        data["is_evening"] = (
            data["Publication_Time"]
            .isin(["Evening", "Night"])
            .astype(int)
        )

    # Sentiment features
    if "Episode_Sentiment" in data.columns:
        sentiment_map = {"Positive": 1, "Neutral": 0, "Negative": -1}
        data["sentiment_numeric"] = data["Episode_Sentiment"].map(
            sentiment_map
        )

    # ---- 3. Ensure categorical features are non-null ----

    cat_features = [
        "Podcast_Name",
        "Genre",
        "Publication_Day",
        "Publication_Time",
        "Episode_Sentiment",
        "episode_length_group",
    ]

    # Final check: make sure all categorical features have no NaNs
    for cat_feat in cat_features:
        if cat_feat in data.columns and data[cat_feat].isna().any():
            print(
                f"Warning: Feature {cat_feat} contains {data[cat_feat].isna().sum()} NaN values"
            )
            data[cat_feat] = (
                data[cat_feat].fillna("Unknown").astype(str)
            )

    return data, cat_features

# Preprocess the data
processed_train, cat_features = preprocess_data(train_df, is_train=True)
processed_test, _ = preprocess_data(test_df, is_train=False)

processed_train.dropna(subset=["Listening_Time_minutes"], inplace=True)


# Define the dataset class
class PodcastDataset(Dataset):
    def __init__(self, X_num, X_cat, y=None):
        self.X_num = X_num
        self.X_cat = X_cat
        self.y = y

    def __len__(self):
        return len(self.X_num)

    def __getitem__(self, idx):
        if self.y is not None:
            return {
                "numeric": torch.FloatTensor(self.X_num[idx]),
                "categorical": torch.LongTensor(self.X_cat[idx]),
                "target": torch.FloatTensor([self.y[idx]]),
            }
        else:
            return {
                "numeric": torch.FloatTensor(self.X_num[idx]),
                "categorical": torch.LongTensor(self.X_cat[idx]),
            }


# Define the neural network model
class TabularNN(nn.Module):
    def __init__(
        self,
        num_features,
        cat_features,
        cat_dims,
        embed_dims,
        hidden_sizes=[128, 64, 32],
    ):
        super().__init__()

        # Classification features embedding
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(cat_dims[i], embed_dims[i])
                for i in range(len(cat_features))
            ]
        )

        # Calculating the sum of embedding dimensions
        self.embed_dim_sum = sum(embed_dims)

        # Fully connected layers
        input_size = num_features + self.embed_dim_sum
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.BatchNorm1d(hidden_sizes[0]),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.BatchNorm1d(hidden_sizes[1]),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_sizes[1], hidden_sizes[2]),
            nn.BatchNorm1d(hidden_sizes[2]),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_sizes[2], 1),
        )

    def forward(self, x_num, x_cat):
        # Process numerical features
        embeddings = [
            self.embeddings[i](x_cat[:, i])
            for i in range(len(self.embeddings))
        ]
        x_cat_embed = torch.cat(embeddings, 1)

        # Concatenate numerical and categorical features
        x = torch.cat([x_num, x_cat_embed], 1)

        # Pass through the fully connected layers
        return self.layers(x)


# Data preprocessing for neural network
def preprocess_for_nn(train_df, test_df, cat_features, num_features):
    # Copy dataframes to avoid modifying the original ones
    train = train_df.copy()
    test = test_df.copy()

    # Scale numerical features
    scaler = StandardScaler()
    train[num_features] = scaler.fit_transform(train[num_features])
    test[num_features] = scaler.transform(test[num_features])

    # Encoding categorical features
    label_encoders = {}
    X_cat_train = np.zeros((len(train), len(cat_features)), dtype=int)
    X_cat_test = np.zeros((len(test), len(cat_features)), dtype=int)
    cat_dims = []

    for i, col in enumerate(cat_features):
        le = LabelEncoder()
        train[col] = train[col].fillna("Unknown")
        test[col] = test[col].fillna("Unknown")

        # Fit label encoder on both train and test data
        all_values = pd.concat([train[col], test[col]]).unique()
        le.fit(all_values)

        X_cat_train[:, i] = le.transform(train[col])
        X_cat_test[:, i] = le.transform(test[col])

        cat_dims.append(len(le.classes_))
        label_encoders[col] = le

    # Prepare numerical features
    X_num_train = train[num_features].values
    X_num_test = test[num_features].values
    y_train = train["Listening_Time_minutes"].values

    return (
        X_num_train,
        X_cat_train,
        X_num_test,
        X_cat_test,
        y_train,
        cat_dims,
    )


# Train the model with cross-validation
def train_model_with_cv(
    train_df, test_df, cat_features, num_features, n_folds=5
):
    # Preprocess data for neural network
    (
        X_num_train,
        X_cat_train,
        X_num_test,
        X_cat_test,
        y_train,
        cat_dims,
    ) = preprocess_for_nn(train_df, test_df, cat_features, num_features)

    # Calculate embedding dimensions
    embed_dims = [min(max(int(dim**0.25), 2), 10) for dim in cat_dims]

    # Set up KFold cross-validation
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    oof_predictions = np.zeros(len(train_df))
    test_predictions = np.zeros(len(test_df))
    fold_scores = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_num_train)):
        print(f"Training fold {fold+1}/{n_folds}")

        train_dataset = PodcastDataset(
            X_num_train[train_idx],
            X_cat_train[train_idx],
            y_train[train_idx],
        )
        val_dataset = PodcastDataset(
            X_num_train[val_idx], X_cat_train[val_idx], y_train[val_idx]
        )

        train_loader = DataLoader(
            train_dataset, batch_size=1024, shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=1024, shuffle=False
        )

        # Instantiate the model
        model = TabularNN(
            num_features=len(num_features),
            cat_features=cat_features,
            cat_dims=cat_dims,
            embed_dims=embed_dims,
        ).to(device)

        # Define optimizer, scheduler, and loss function
        optimizer = optim.Adam(
            model.parameters(), lr=0.001, weight_decay=1e-5
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3
        )
        criterion = nn.MSELoss()

        # Training loop and validation
        best_val_rmse = float("inf")
        patience, early_stop_counter = 10, 0

        # 100 epochs
        for epoch in range(100):
            model.train()
            train_loss = 0

            for batch in train_loader:
                x_num = batch["numeric"].to(device)
                x_cat = batch["categorical"].to(device)
                y = batch["target"].to(device)

                optimizer.zero_grad()
                outputs = model(x_num, x_cat)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * x_num.size(0)

            # Evaluate on validation set
            model.eval()
            val_loss = 0
            val_preds = []
            val_targets = []

            with torch.no_grad():
                for batch in val_loader:
                    x_num = batch["numeric"].to(device)
                    x_cat = batch["categorical"].to(device)
                    y = batch["target"].to(device)

                    outputs = model(x_num, x_cat)
                    loss = criterion(outputs, y)
                    val_loss += loss.item() * x_num.size(0)

                    val_preds.extend(outputs.cpu().numpy().flatten())
                    val_targets.extend(y.cpu().numpy().flatten())

            train_loss /= len(train_idx)
            val_loss /= len(val_idx)
            val_rmse = np.sqrt(
                mean_squared_error(val_targets, val_preds)
            )

            scheduler.step(val_rmse)

            print(
                f"Epoch {epoch+1}, Train Loss: {train_loss:.6f}, Val RMSE: {val_rmse:.6f}"
            )

            # Check for best validation score
            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                torch.save(
                    model.state_dict(), f"best_model_fold{fold}.pth"
                )
                early_stop_counter = 0
            else:
                early_stop_counter += 1

            # 早停
            if early_stop_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        # Load the best model for this fold
        model.load_state_dict(torch.load(f"best_model_fold{fold}.pth"))

        # Predict on validation set
        model.eval()
        val_predictions = []

        with torch.no_grad():
            val_dataset = PodcastDataset(
                X_num_train[val_idx], X_cat_train[val_idx]
            )
            val_loader = DataLoader(
                val_dataset, batch_size=1024, shuffle=False
            )

            for batch in val_loader:
                x_num = batch["numeric"].to(device)
                x_cat = batch["categorical"].to(device)

                outputs = model(x_num, x_cat)
                val_predictions.extend(outputs.cpu().numpy().flatten())

        oof_predictions[val_idx] = val_predictions
        fold_score = np.sqrt(
            mean_squared_error(y_train[val_idx], val_predictions)
        )
        fold_scores.append(fold_score)
        print(f"Fold {fold+1} RMSE: {fold_score:.6f}")

        # Predict on test set
        test_dataset = PodcastDataset(X_num_test, X_cat_test)
        test_loader = DataLoader(
            test_dataset, batch_size=1024, shuffle=False
        )
        test_fold_preds = []

        with torch.no_grad():
            for batch in test_loader:
                x_num = batch["numeric"].to(device)
                x_cat = batch["categorical"].to(device)

                outputs = model(x_num, x_cat)
                test_fold_preds.extend(outputs.cpu().numpy().flatten())

        test_predictions += np.array(test_fold_preds) / n_folds

    # Calculate overall validation score
    overall_score = np.sqrt(
        mean_squared_error(y_train, oof_predictions)
    )
    print(f"Overall validation RMSE: {overall_score:.6f}")

    return test_predictions, oof_predictions, overall_score


# Add the target variable to the processed train DataFrame
exclude_cols = cat_features + [
    "id",
    "Episode_Title",
    "Listening_Time_minutes",
]
num_features = [
    col for col in processed_train.columns if col not in exclude_cols
]

# Train the model with cross-validation
test_predictions, oof_predictions, cv_score = train_model_with_cv(
    processed_train, processed_test, cat_features, num_features
)


submission = pd.read_csv("/kaggle/input/playground-series-s5e4/sample_submission.csv")
submission["Listening_Time_minutes"] = test_predictions

submission.to_csv("submission_nn.csv", index=False)

print(f"Final cross-validation RMSE: {cv_score:.6f}")
print("Submission file created.")

