import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report

dataframes = {}

# Load datasets
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        if filename.endswith(".csv"):
            file_path = os.path.join(dirname, filename)
            file_key = filename.replace(".csv", "")

            # Try multiple encodings
            try:
                dataframes[file_key] = pd.read_csv(file_path, encoding="utf-8")
            except UnicodeDecodeError:
                try:
                    dataframes[file_key] = pd.read_csv(file_path, encoding="ISO-8859-1")
                except UnicodeDecodeError:
                    dataframes[file_key] = pd.read_csv(file_path, encoding="latin1")

# Handle missing values
def handle_missing_values(df, threshold=0.4):
    missing_ratio = df.isnull().sum() / len(df)
    cols_to_drop = missing_ratio[missing_ratio > threshold].index
    df = df.drop(columns=cols_to_drop)

    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype == "object":
                df[col] = df[col].fillna(df[col].mode()[0])
            elif pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
    return df

cleaned_dataframes = {name: handle_missing_values(df) for name, df in dataframes.items()}

# Encode categorical columns
def encode_categorical_columns(df):
    for col in df.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
    return df

processed_dataframes = {name: encode_categorical_columns(df) for name, df in cleaned_dataframes.items()}


for name, df in processed_dataframes.items():
    print(f"Exploring Dataset: {name}")

    # 1. Display basic statistics
    print(df.describe())

    # 2. Correlation heatmap (Only for numerical features)
    plt.figure(figsize=(10,6))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title(f"Correlation Heatmap: {name}")
    plt.show()

    # 3. Boxplot for detecting outliers
    for col in df.select_dtypes(include=[np.number]).columns:
        plt.figure(figsize=(8, 4))
        sns.boxplot(x=df[col])
        plt.title(f"Boxplot: {col} ({name})")
        plt.show()

# Feature Engineering
def feature_engineering(df):
    if 'wscore' in df.columns and 'lscore' in df.columns:
        df['total_points'] = df['wscore'] + df['lscore']
        df['point_diff'] = df['wscore'] - df['lscore']
    df['numot'] = df['numot'].fillna(0)
    return df

df = processed_dataframes['MNCAATourneyDetailedResults']
df.columns = df.columns.str.lower()
df = feature_engineering(df)

# Create a histogram for a specific numerical column (e.g., 'total_points')
plt.figure(figsize=(10, 6))
sns.histplot(df['total_points'], bins=30, kde=True, color='skyblue', edgecolor='black')
plt.title('Histogram of Total Points')
plt.xlabel('Total Points')
plt.ylabel('Frequency')
plt.show()

# Create a boxplot for point differences (win margins)
plt.figure(figsize=(8, 6))
sns.boxplot(x=df['point_diff'], color='lightgreen',
            boxprops=dict(edgecolor='black'),  # Adjust the box border color
            flierprops=dict(markerfacecolor='red', marker='o', markersize=5))  # Customize outlier points
plt.title('Boxplot of Point Differences')
plt.xlabel('Point Difference (Win Margin)')
plt.show()

# Calculate standard deviation for relevant numerical columns
std_devs = df[['wscore', 'lscore', 'total_points', 'point_diff']].std()

# Display standard deviations
print("Standard Deviations of Relevant Columns:")
print(std_devs)

# Visualize standard deviations using a bar chart
plt.figure(figsize=(8, 6))
std_devs.plot(kind='bar', color='lightblue', edgecolor='black')
plt.title('Standard Deviation Analysis of Key Features')
plt.ylabel('Standard Deviation')
plt.xlabel('Features')
plt.xticks(rotation=45)
plt.show()


# Prepare dataset
df['winner'] = df['wteamid'].astype(int)
X = df[['wteamid', 'lteamid', 'daynum', 'numot', 'total_points', 'point_diff']].copy()
y = df['winner']

from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np
import pandas as pd

# Standardize numerical features before training
X[['daynum', 'numot', 'total_points', 'point_diff']] = X[['daynum', 'numot', 'total_points', 'point_diff']].astype(float)
scaler = StandardScaler()
X.loc[:, ['daynum', 'numot', 'total_points', 'point_diff']] = scaler.fit_transform(X[['daynum', 'numot', 'total_points', 'point_diff']])

# Encode `y` so that classes are numbered from 0 to N-1
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Remove rare classes (teams with ≤ 2 samples)
class_counts = pd.Series(y).value_counts()
valid_classes = class_counts[class_counts > 2].index  # Keep teams with at least 3 samples
X = X[pd.Series(y).isin(valid_classes)]
y = pd.Series(y)[pd.Series(y).isin(valid_classes)].values  # Ensure correct format

# Reapply Label Encoding After Filtering to Ensure Sequential Labels
label_encoder = LabelEncoder()  # Reinitialize LabelEncoder
y = label_encoder.fit_transform(y)  # Encode again to ensure labels are continuous (0, 1, 2, ..., N-1)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

print("✅ Data preprocessing successful. Ready for model training.")


# Model Training with Hyperparameter Tuning
xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
params = {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1], 'max_depth': [3, 5]}
grid_search = GridSearchCV(xgb, params, cv=3, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Evaluate model
y_pred = grid_search.best_estimator_.predict(X_test)
print(f"Best Model Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred))

# Save Model and Scaler
with open("model.pkl", "wb") as model_file:
    pickle.dump(grid_search.best_estimator_, model_file)
with open("scaler.pkl", "wb") as scaler_file:
    pickle.dump(scaler, scaler_file)

print("✅ Model and Scaler saved successfully!")

import itertools
import pandas as pd
import pickle
import numpy as np

# Load trained model and scaler
with open("model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

# Define team ID ranges
team_range_1 = range(1101, 1481)  # Teams 1101 to 1480
team_range_2 = range(3101, 3481)  # Teams 3101 to 3480

# Generate all possible matchups within each group
matchups_1 = list(itertools.combinations(team_range_1, 2))
matchups_2 = list(itertools.combinations(team_range_2, 2))

# Combine all matchups
all_matchups = matchups_1 + matchups_2

# Create a DataFrame for the matchups
submission_df = pd.DataFrame(all_matchups, columns=["wteamid", "lteamid"])

# Add required features (assuming neutral values)
submission_df["daynum"] = 50  # Example fixed value, adjust as needed
submission_df["numot"] = 0
submission_df["total_points"] = 140  # Example total points, adjust if necessary
submission_df["point_diff"] = 0  # Neutral point difference

# Scale numerical features
submission_df[["daynum", "numot", "total_points", "point_diff"]] = scaler.transform(
    submission_df[["daynum", "numot", "total_points", "point_diff"]]
)

raw_probs = model.predict_proba(submission_df)
print(raw_probs[:10])  # Print first 10 predictions

print(model.classes_)  # Check class order

# Predict the probability of the first team winning
submission_df["Pred"] = model.predict_proba(submission_df)[:, np.argmax(model.classes_)]  # Pick highest class probability

# submission_df["Pred"] = (model.predict_proba(submission_df)[:, 1] - 0.5) * 2  # Scaled to [-1,1]
print(submission_df["Pred"])
# Create the ID column
submission_df["ID"] = submission_df.apply(lambda row: f"2025_{int(row['wteamid'])}_{int(row['lteamid'])}", axis=1)

# Select only the required columns
final_submission = submission_df[["ID", "Pred"]]

# Save to CSV
final_submission.to_csv("submission.csv", index=False)

print("✅ Submission file saved as submission.csv!")




