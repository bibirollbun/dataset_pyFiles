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


train_path = "/kaggle/input/playground-series-s5e4/train.csv"  
test_path = "/kaggle/input/playground-series-s5e4/test.csv" 


train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)



train_df.head()
test_df.head()


train_df.head()


print("Train Data:")
print(train_df.info())


print("Test Data:")
print(test_df.info())


train_df.describe()


test_df.describe()



target_column = "Listening_Time_minutes"
y_train = train_df[target_column]

X_train = train_df.drop(columns=[target_column])
X_test = test_df


y_train.head()


numerical_features = X_train.select_dtypes(include=['number']).columns.tolist()
categorical_features =X_test.select_dtypes(exclude=['number']).columns.tolist()
print("Numerical Features:", numerical_features)  
print("Categorical Features:", categorical_features)


# Check for missing values in train dataset
print("Missing values in train dataset:")
print(train_df.isnull().sum())

# Check for missing values in test dataset
print("\nMissing values in test dataset:")
print(test_df.isnull().sum())



train_df['Episode_Length_minutes'].fillna(train_df['Episode_Length_minutes'].median(), inplace=True)
train_df['Guest_Popularity_percentage'].fillna(train_df['Guest_Popularity_percentage'].median(), inplace=True)

test_df['Episode_Length_minutes'].fillna(test_df['Episode_Length_minutes'].median(), inplace=True)
test_df['Guest_Popularity_percentage'].fillna(test_df['Guest_Popularity_percentage'].median(), inplace=True)



train_df['Number_of_Ads'].fillna(train_df['Number_of_Ads'].mode()[0], inplace=True)



print(train_df.isnull().sum())
print(test_df.isnull().sum())


import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Define a custom color palette
custom_palette = ['#3498db', '#e74c3c', '#2ecc71']

# Ensure `X_train` and `X_test` exist before modification
X_train = X_train.copy()
X_test = X_test.copy()

# Add Source column
X_train['Source'] = 'Train'
X_test['Source'] = 'Test'

# Add target to X_train
X_train['Listening_Time_minutes'] = y_train

# Combine train and test for visualization
combined_df = pd.concat([X_train, X_test], ignore_index=True)

def generate_numerical_feature_visualizations(feature_name):
    """Generate box plot and histogram for numerical features."""
    sns.set(style='whitegrid')
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Boxplot
    sns.boxplot(data=combined_df, x=feature_name, y="Source", palette=custom_palette, ax=axes[0])
    axes[0].set_xlabel(feature_name)
    axes[0].set_title(f"Box Plot for {feature_name} Across Datasets")

    # Histogram
    sns.histplot(data=X_train, x=feature_name, color=custom_palette[0], kde=True, bins=30, label="Train", alpha=0.6, ax=axes[1])
    sns.histplot(data=X_test, x=feature_name, color=custom_palette[1], kde=True, bins=30, label="Test", alpha=0.6, ax=axes[1])
    axes[1].set_xlabel(feature_name)
    axes[1].set_ylabel("Frequency")
    axes[1].set_title(f"Histogram for {feature_name} (Train vs Test)")
    axes[1].legend(title="Dataset")

    plt.tight_layout()
    plt.show()

def generate_categorical_feature_visualizations(feature_name):
    """Generate box plot for categorical features vs Listening Time."""
    sns.set(style='whitegrid')
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=combined_df, x=feature_name, y="Listening_Time_minutes", palette=custom_palette)
    plt.xlabel(feature_name)
    plt.ylabel("Listening Time (Minutes)")
    plt.title(f"Box Plot for {feature_name} vs Target")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Ensure `numerical_features` and `categorical_features` are defined
numerical_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X_train.select_dtypes(include=['object']).columns.tolist()

# Remove 'Listening_Time_minutes' from numerical features (it is the target variable)
if 'Listening_Time_minutes' in numerical_features:
    numerical_features.remove('Listening_Time_minutes')

# Generate visualizations
for feature in numerical_features:
    generate_numerical_feature_visualizations(feature)

for feature in categorical_features:
    generate_categorical_feature_visualizations(feature)

# Drop 'Source' column after visualization
X_train.drop(columns=['Source', 'Listening_Time_minutes'], inplace=True)
X_test.drop(columns=['Source'], inplace=True)



numerical_train = X_train[numerical_features]  
numerical_test = X_test[numerical_features]   

train_with_target = numerical_train.copy()
train_with_target['Listening_Time_minutes'] = y_train

test_with_target = numerical_test.copy()

corr_train = train_with_target.corr()
corr_test = test_with_target.corr()

mask_train = np.triu(np.ones_like(corr_train, dtype=bool))
mask_test = np.triu(np.ones_like(corr_test, dtype=bool))

annot_kws = {"size": 16, "rotation": 45}  

plt.figure(figsize=(24, 24))  

plt.subplot(2, 1, 1) 
sns.heatmap(corr_train, mask=mask_train, cmap='viridis', annot=True,
            square=True, linewidths=.5, xticklabels=1, yticklabels=1, annot_kws=annot_kws)
plt.title('Correlation Heatmap - Train Data', fontsize=24)

plt.subplot(2, 1, 2)  
sns.heatmap(corr_test, mask=mask_test, cmap='viridis', annot=True,
            square=True, linewidths=.5, xticklabels=1, yticklabels=1, annot_kws=annot_kws)
plt.title('Correlation Heatmap - Test Data', fontsize=24)

plt.tight_layout()

plt.show()


import numpy as np


# Select numerical features from train & test sets
numerical_train = X_train[numerical_features]
numerical_test = X_test[numerical_features]

# Create copies to avoid modifying original DataFrames
train_with_target = numerical_train.copy()
train_with_target['Listening_Time_minutes'] = y_train  # Add target column

# Compute correlation matrices
corr_train = train_with_target.corr()
corr_test = numerical_test.corr()  # Test set has no target column

# Create masks for upper triangle of the heatmaps
mask_train = np.triu(np.ones_like(corr_train, dtype=bool))
mask_test = np.triu(np.ones_like(corr_test, dtype=bool))

# Heatmap styling
annot_kws = {"size": 12}  

plt.figure(figsize=(18, 16))  # Adjusted size

# Train correlation heatmap
plt.subplot(2, 1, 1)
sns.heatmap(corr_train, mask=mask_train, cmap='viridis', annot=True,
            square=True, linewidths=0.5, xticklabels=True, yticklabels=True, annot_kws=annot_kws)
plt.title('Correlation Heatmap - Train Data', fontsize=18)

# Test correlation heatmap
plt.subplot(2, 1, 2)
sns.heatmap(corr_test, mask=mask_test, cmap='viridis', annot=True,
            square=True, linewidths=0.5, xticklabels=True, yticklabels=True, annot_kws=annot_kws)
plt.title('Correlation Heatmap - Test Data', fontsize=18)

plt.tight_layout()
plt.show()

# -------------------------------------------
# ðŸ”¹ Correlation of Features with Target Only
# -------------------------------------------

# Compute correlation with target variable
corr_train_target = train_with_target.corr()[['Listening_Time_minutes']].T  

plt.figure(figsize=(12, 3))  
sns.heatmap(corr_train_target, cmap='viridis', annot=True,
            square=False, linewidths=0.5, annot_kws=annot_kws,
            cbar=False)

plt.xticks(rotation=45, ha="right")  
plt.title('Feature Correlation with Target (Train Data)')
plt.yticks(rotation=0) 
plt.show()




categorical_pie_features = ["Genre", "Publication_Day", "Publication_Time", "Episode_Sentiment"]

colors = ['#3498db', '#e74c3c', '#2ecc71', '#f1c40f', '#9b59b6', '#1abc9c', '#ff5733']

plt.figure(figsize=(16, 12))

for i, feature in enumerate(categorical_pie_features, 1):
    plt.subplot(2, 2, i)  
    counts = X_train[feature].value_counts()
    
    wedges, texts, autotexts = plt.pie(
        counts, labels=counts.index, autopct='%1.1f%%', colors=colors[:len(counts)], 
        startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'}
    )
    
    for autotext in autotexts:
        autotext.set_fontweight('bold')
        autotext.set_fontsize(14)  
    plt.title(f"Distribution of {feature}", fontsize=14, fontweight='bold')

plt.tight_layout()
plt.show()


X_train.head()


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import pandas as pd
from catboost import CatBoostRegressor

# Define features
numerical_features = ['Episode_Length_minutes', 'Host_Popularity_percentage', 
                      'Guest_Popularity_percentage', 'Number_of_Ads']
categorical_features = ['Podcast_Name', 'Episode_Title', 'Genre', 
                        'Publication_Day', 'Publication_Time', 'Episode_Sentiment']

# Drop 'id' column if present
if 'id' in X_train.columns:
    X_train = X_train.drop(columns=['id'])
if 'id' in X_test.columns:
    X_test = X_test.drop(columns=['id'])

# Convert categorical features to string for CatBoost
for col in categorical_features:
    X_train[col] = X_train[col].astype(str)
    X_test[col] = X_test[col].astype(str)

# Preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='median'), numerical_features),
        ('cat', 'passthrough', categorical_features)
    ])

# Split data
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Combine preprocessing and model in pipeline
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', CatBoostRegressor(
        iterations=1000,
        learning_rate=0.01,
        depth=6,
        random_state=42,
        early_stopping_rounds=20,
        verbose=0
    ))
])

# Get the column indices of categorical features after transformation
cat_feature_indices = list(range(len(numerical_features), len(numerical_features) + len(categorical_features)))

# Fit model
model_pipeline.fit(X_tr, y_tr,
                   regressor__cat_features=cat_feature_indices)

# Predict and evaluate
y_pred = model_pipeline.predict(X_val)
mae = mean_absolute_error(y_val, y_pred)
print(f"Validation MAE: {mae:.4f}")

# Predict on test data
test_preds = model_pipeline.predict(X_test)

# Prepare submission
submission = pd.DataFrame({
    "id": test_df["id"],
    "Listening_Time_minutes": test_preds
})

submission.to_csv("submission.csv", index=False)
print("Submission file created!")





