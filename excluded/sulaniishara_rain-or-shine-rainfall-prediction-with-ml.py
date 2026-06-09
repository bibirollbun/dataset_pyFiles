!pip install tabpfn


# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# Data manipulation and analysis
import pandas as pd
import numpy as np

# Visualization libraries
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D
import seaborn as sns

# Display data in Jupyter notebooks
from IPython.display import display

# Machine learning libraries
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Classifier models
from tabpfn import TabPFNClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Evaluation metrics
from sklearn.metrics import (
    accuracy_score, roc_auc_score, classification_report,
    confusion_matrix, roc_curve, auc
)


# Load the datasets
train_data = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv',index_col='id')
test_data = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv',index_col='id')
sample_data = pd.read_csv('/kaggle/input/playground-series-s5e3/sample_submission.csv')
original_data = pd.read_csv('/kaggle/input/rainfall-prediction-using-machine-learning/Rainfall.csv')


# Verify shapes
print("Train Data Shape:", train_data.shape)
print("Original Data Shape:", original_data.shape)
print("Test Data Shape:", test_data.shape)


# Display few rows of each dataset
print("Train Data Preview:")
display(train_data.tail())

print("\nOriginal Data Preview:")
display(original_data.tail())

print("\nTest Data Preview:")
display(test_data.head())


# Display information about the DataFrames
print("\nTrain Data Info:")
train_data.info()

print("\nOriginal Data Info:")
original_data.info()

print("\nTest Data Info:")
test_data.info()


# Remove spaces from column names
original_data.columns = original_data.columns.str.strip()

print("\nUpdated Column Names:")
print(original_data.columns)


# Correct spelling inconsistency in train_data
train_data = train_data.rename(columns={'temparature': 'temperature'})

# Correct spelling inconsistency in original_data
original_data = original_data.rename(columns={'temparature': 'temperature'})

# Correct spelling inconsistency in test_data
test_data = test_data.rename(columns={'temparature': 'temperature'})


# Reorder columns in original_data to match train_data
original_data = original_data.reindex(columns=train_data.columns)
print("Original Data Columns After Reordering:")
print(original_data.columns)


# Descriptive statistics for numerical columns
print("\nTrain Data Describe:")
display(train_data.describe().T.style.background_gradient(cmap='BrBG'))

print("\nOriginal Data Describe:")
display(original_data.describe().T.style.background_gradient(cmap='BrBG'))

print("\nTest Data Describe:")
display(test_data.describe().T.style.background_gradient(cmap='BrBG'))


# Function to create a summary table for missing values and data types
def missing_values_summary(df):
    missing_count = df.isnull().sum()
    missing_percentage = 100 * missing_count / len(df)
    data_types = df.dtypes
    return pd.DataFrame({
        'Missing Values Count': missing_count,
        'Percentage (%)': missing_percentage,
        'Data Type': data_types
    })

# Create summary tables
train_summary = missing_values_summary(train_data)
test_summary = missing_values_summary(test_data)
original_summary = missing_values_summary(original_data)

print("Train Dataset Summary:")
display(train_summary)

print("Original Dataset Summary:")
display(original_summary)

print("\nTest Dataset Summary:")
display(test_summary)


# Check for duplicated rows
print("\nDuplicate Rows in Train Data:", train_data.duplicated().sum())
print("\nDuplicate Rows in Original Data:", original_data.duplicated().sum())
print("\nDuplicate Rows in Test Data:", test_data.duplicated().sum())


# Inspect unique values in 'rainfall' column of original_data
print("Unique Values in 'rainfall' Column:")
print(original_data['rainfall'].unique())

# Convert 'rainfall' to binary integer type (0/1)
# Assuming 'yes' indicates rainfall and 'no' indicates no rainfall
original_data['rainfall'] = (original_data['rainfall'] == 'yes').astype(int)

# Verify the data type after conversion
print("\nData Type of 'rainfall' After Conversion:")
print(original_data['rainfall'].dtype)



# Set target variable
target_variable = 'rainfall'


custom_palette = sns.color_palette("BrBG", 2)

def plot_target_distribution(data, target_variable, title_suffix="", custom_palette=None):
    if custom_palette is None:
        custom_palette = sns.color_palette("BrBG", 2)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    sns.countplot(y=target_variable, data=data, ax=axes[0], palette=custom_palette)
    axes[0].set_title(f'Distribution of {target_variable} in {title_suffix}')
    axes[0].set_xlabel('Count')
    axes[0].set_ylabel(target_variable)

    for p in axes[0].patches:
        axes[0].annotate(f'{int(p.get_width())}', 
                         (p.get_width(), p.get_y() + p.get_height() / 2), 
                         ha='left', va='center', 
                         color='black', fontsize=10)

    axes[0].set_axisbelow(True)
    axes[0].grid(axis='x', color='gray', linestyle='--', linewidth=0.7)  
    sns.despine(left=True, bottom=True)

    rainfall_counts = data[target_variable].value_counts()
    wedges, texts, autotexts = axes[1].pie(
        rainfall_counts, 
        labels=rainfall_counts.index, 
        autopct='%1.1f%%', 
        startangle=90,
        colors=custom_palette
    )
    centre_circle = plt.Circle((0, 0), 0.70, fc='white')
    fig.gca().add_artist(centre_circle)
    axes[1].set_title(f'Percentage of {target_variable} Distribution in {title_suffix}')
    axes[1].axis('equal')

    plt.tight_layout()
    plt.show()


plot_target_distribution(train_data, 'rainfall', title_suffix="Train Data", custom_palette=custom_palette)
plot_target_distribution(original_data, 'rainfall', title_suffix="Original Data", custom_palette=custom_palette)



cmap = plt.get_cmap('BrBG')
colors = [cmap(0.8), cmap(0.3), cmap(0)]

# List of numerical features (excluding 'rainfall')
numerical_features = ['day', 'pressure', 'maxtemp', 'temperature', 'mintemp', 
                      'dewpoint', 'humidity', 'cloud', 'sunshine', 
                      'winddirection', 'windspeed']

fig, axes = plt.subplots(len(numerical_features), 2, figsize=(12, len(numerical_features) * 4))

for i, feature in enumerate(numerical_features):
    # Histogram
    sns.histplot(train_data[feature], color=colors[0], label='Train Data', bins=20, kde=True, ax=axes[i, 0])
    sns.histplot(test_data[feature], color=colors[1], label='Test Data', bins=20, kde=True, ax=axes[i, 0])
    sns.histplot(original_data[feature], color=colors[2], label='Original Data', bins=20, kde=True, ax=axes[i, 0])
    axes[i, 0].set_title(f'Histogram of {feature}')
    axes[i, 0].legend()
    axes[i, 0].grid(color='gray', linestyle='--', linewidth=0.7)

    # Horizontal Boxplot
    sns.boxplot(data=[train_data[feature], test_data[feature], original_data[feature]], 
                palette=colors, orient='h', ax=axes[i, 1])
    axes[i, 1].set_title(f'Horizontal Boxplot of {feature}')
    axes[i, 1].set_yticklabels(['Train Data', 'Test Data', 'Original Data'])
    axes[i, 1].grid(axis='x', color='gray', linestyle='--', linewidth=0.7)

plt.tight_layout()
plt.show()



custom_palette = sns.color_palette("BrBG_r", 35)
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

train_order = train_data['winddirection'].value_counts().index
test_order = test_data['winddirection'].value_counts().index
original_order = original_data['winddirection'].value_counts().index

sns.countplot(y='winddirection', data=train_data, ax=axes[0], palette=custom_palette, order=train_order)
axes[0].set_title('Train Data Wind Direction')
axes[0].set_xlabel('Count')
axes[0].set_ylabel('Wind Direction')

sns.countplot(y='winddirection', data=test_data, ax=axes[1], palette=custom_palette, order=test_order)
axes[1].set_title('Test Data Wind Direction')
axes[1].set_xlabel('Count')
axes[1].set_ylabel('Wind Direction')

sns.countplot(y='winddirection', data=original_data, ax=axes[2], palette=custom_palette, order=original_order)
axes[2].set_title('Original Data Wind Direction')
axes[2].set_xlabel('Count')
axes[2].set_ylabel('Wind Direction')

for ax in axes:
    for p in ax.patches:
        ax.annotate(f'{int(p.get_width())}', 
                     (p.get_width(), p.get_y() + p.get_height() / 2), 
                     ha='left', va='center', 
                     color='black', fontsize=10)

for ax in axes:
    ax.set_axisbelow(True)  
    ax.grid(axis='x', color='gray', linestyle='--', linewidth=0.7)  
    sns.despine(left=True, bottom=True)

plt.tight_layout()
plt.show()



cmap = plt.get_cmap('BrBG')
colors = [cmap(0.8), cmap(0.3), cmap(0)]

# List of numerical features (excluding 'day' and 'rainfall')
numerical_features = ['pressure', 'maxtemp', 'temperature', 'mintemp', 
                      'dewpoint', 'humidity', 'cloud', 'sunshine', 
                      'winddirection', 'windspeed']

fig, axes = plt.subplots(len(numerical_features), 3, figsize=(20, len(numerical_features) * 5))

for i, feature in enumerate(numerical_features):
    # Train Data Scatter Plot
    sns.scatterplot(x='day', y=feature, color=colors[0], data=train_data, ax=axes[i, 0])
    axes[i, 0].set_title(f'{feature} vs day - Train Data')
    axes[i, 0].grid(color='gray', linestyle='--', linewidth=0.7)

    # Test Data Scatter Plot
    sns.scatterplot(x='day', y=feature, color=colors[1], data=test_data, ax=axes[i, 1])
    axes[i, 1].set_title(f'{feature} vs day - Test Data')
    axes[i, 1].grid(color='gray', linestyle='--', linewidth=0.7)

    # original Data Scatter Plot
    sns.scatterplot(x='day', y=feature, color=colors[2], data=original_data, ax=axes[i, 2])
    axes[i, 2].set_title(f'{feature} vs day - Original Data')
    axes[i, 2].grid(color='gray', linestyle='--', linewidth=0.7)

plt.tight_layout()
plt.show()


# Define colormap and colors
cmap = plt.get_cmap('BrBG')
colors = [cmap(0.2), cmap(0.8)]

# Create subplots
fig, axes = plt.subplots(len(numerical_features), 2, figsize=(20, len(numerical_features) * 5))

for i, feature in enumerate(numerical_features):
    # Train Data Scatter Plot
    sns.scatterplot(x='day', y=feature, hue='rainfall', palette=colors, data=train_data, ax=axes[i, 0])
    axes[i, 0].set_title(f'{feature} vs day by Rainfall - Train Data')
    axes[i, 0].legend(title='Rainfall', loc='upper right')  
    axes[i, 0].grid(color='gray', linestyle='--', linewidth=0.7)

    # Train Data Histogram
    sns.histplot(x=feature, hue='rainfall', palette=colors, data=train_data, ax=axes[i, 1], kde=True)
    axes[i, 1].set_title(f'{feature} Distribution by Rainfall - Train Data')
    
    # Fix legend for histogram
    handles = [plt.Line2D([0], [0], color=colors[0], lw=4, label='No Rain'),
               plt.Line2D([0], [0], color=colors[1], lw=4, label='Rain')]
    axes[i, 1].legend(handles=handles, title='Rainfall', loc='upper right')  
    
    axes[i, 1].grid(color='gray', linestyle='--', linewidth=0.7)

plt.tight_layout()
plt.show()


fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Dewpoint vs Humidity
sns.kdeplot(
    x="dewpoint", y="humidity", hue="rainfall", 
    palette=colors, data=train_data, ax=axes[0]
)
axes[0].set_title('Dewpoint vs Humidity by Rainfall')
axes[0].grid(color='gray', linestyle='--', linewidth=0.7)

# Cloud vs Humidity
sns.kdeplot(
    x="cloud", y="humidity", hue="rainfall", 
    palette=colors, data=train_data, ax=axes[1]
)
axes[1].set_title('Cloud vs Humidity by Rainfall')
axes[1].grid(color='gray', linestyle='--', linewidth=0.7)

plt.tight_layout()
plt.show()



fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Temperature vs Humidity
sns.kdeplot(
    x="temperature", y="humidity", hue="rainfall", 
    palette=colors, data=train_data, ax=axes[0]
)
axes[0].set_title('Temperature vs Humidity by Rainfall')
axes[0].grid(color='gray', linestyle='--', linewidth=0.7)

# Temperature vs Cloud
sns.kdeplot(
    x="temperature", y="cloud", hue="rainfall", 
    palette=colors, data=train_data, ax=axes[1]
)
axes[1].set_title('Temperature vs Cloud by Rainfall')
axes[1].grid(color='gray', linestyle='--', linewidth=0.7)

plt.tight_layout()
plt.show()


fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Wind Direction vs Pressure
sns.kdeplot(
    x="pressure", y="winddirection", hue="rainfall", 
    palette=colors, data=train_data, ax=axes[0]
)
axes[0].set_title('Pressure vs Wind Direction by Rainfall')
axes[0].grid(color='gray', linestyle='--', linewidth=0.7)

# Wind Speed vs Pressure
sns.kdeplot(
    x="pressure", y="windspeed", hue="rainfall", 
    palette=colors, data=train_data, ax=axes[1]
)
axes[1].set_title('Pressure vs Wind Speed by Rainfall')
axes[1].grid(color='gray', linestyle='--', linewidth=0.7)

plt.tight_layout()
plt.show()



cmap = plt.get_cmap('BrBG')
colors = [cmap(0.8), cmap(0.3)]

# List of numerical features (excluding 'rainfall')
numerical_features = ['day', 'pressure', 'maxtemp', 'temperature', 'mintemp', 
                      'dewpoint', 'humidity', 'cloud', 'sunshine', 
                      'winddirection', 'windspeed']

train_data['source'] = 'Train'
test_data['source'] = 'Test'

combined_data = pd.concat([train_data, test_data], ignore_index=True)

fig, axes = plt.subplots(len(numerical_features), 1, figsize=(15, len(numerical_features) * 3))

for i, feature in enumerate(numerical_features):
    sns.lineplot(data=combined_data[combined_data['source'] == 'Train'], 
                 x=combined_data[combined_data['source'] == 'Train'].index, 
                 y=feature, color=colors[0], marker='o', ax=axes[i], label='Train')
    sns.lineplot(data=combined_data[combined_data['source'] == 'Test'], 
                 x=combined_data[combined_data['source'] == 'Test'].index, 
                 y=feature, color=colors[1], marker='o', ax=axes[i], label='Test')
    
    axes[i].set_title(f'{feature} Over Time')
    axes[i].set_xlabel('ID')
    axes[i].set_ylabel(feature)
    axes[i].grid(color='gray', linestyle='--', linewidth=0.7)
    axes[i].legend()

plt.tight_layout()
plt.show()



# Create the 'expected_day' column
train_data['expected_day'] = (train_data.index % 365) + 1

# Identify mislabeled days
train_data['day_mislabelled'] = train_data['day'] != train_data['expected_day']

# Count mislabeled days
mislabeled_count = train_data['day_mislabelled'].sum()
print(f"Total mislabeled days: {mislabeled_count}")

# Print mislabeled days
print("\nMislabeled Days:")
for idx, row in train_data[train_data['day_mislabelled']].iterrows():
    print(f"Index: {idx}, Expected Day: {row['expected_day']}, Actual Day: {row['day']}")



# Calculate differences between consecutive days
day_diffs = train_data['day'].diff()

# Plot day differences with mislabeled points
plt.figure(figsize=(15, 5))
plt.plot(day_diffs, marker='o', linestyle='-', color=colors[0], label="Day Differences")

# Highlight mislabeled points
mislabeled_indices = train_data[train_data['day_mislabelled']].index
plt.scatter(mislabeled_indices, day_diffs.loc[mislabeled_indices], color='red', label="Mislabeled Points", zorder=3)

plt.title('Day Differences with Mislabeled Points')
plt.xlabel('Index')
plt.ylabel('Day Difference')
plt.legend()
plt.grid(color='gray', linestyle='--', linewidth=0.7)
plt.show()


# Visualize mislabeled days
plt.figure(figsize=(15, 5))
plt.plot(train_data.index, train_data['expected_day'], label='Expected Day', marker='s', linestyle='-', color=colors[0])
plt.plot(train_data.index, train_data['day'], label='Actual Day', marker='o', linestyle='-', color=colors[1])

# Highlight mislabeled days
mislabeled_indices = train_data[train_data['day_mislabelled']].index
plt.scatter(mislabeled_indices, train_data.loc[mislabeled_indices, 'day'], color='red', label='Mislabeled Days', zorder=3)

plt.title('Expected vs. Actual Days with Mislabeled Points')
plt.xlabel('Index')
plt.ylabel('Day')
plt.legend()
plt.grid(color='gray', linestyle='--', linewidth=0.7)
plt.show()


cmap = plt.get_cmap('BrBG')
colors = [cmap(0.8), cmap(0.3)]

train_data['source'] = 'Train'
test_data['source'] = 'Test'

combined_data = pd.concat([train_data, test_data], ignore_index=True)

# Identify mislabeled indices in train data
# mislabeled_indices = train_data[train_data['day_mislabelled']].index

fig, axes = plt.subplots(len(numerical_features), 1, figsize=(15, len(numerical_features) * 3))

for i, feature in enumerate(numerical_features):
    sns.lineplot(data=combined_data[combined_data['source'] == 'Train'], 
                 x=combined_data[combined_data['source'] == 'Train'].index, 
                 y=feature, color=colors[0], marker='o', ax=axes[i], label='Train')
    
    # Highlight mislabeled points in train data
    if feature in train_data.columns:
        mislabeled_values = train_data.loc[mislabeled_indices, feature]
        axes[i].scatter(mislabeled_indices, mislabeled_values, color='red', marker='o', zorder=3, label='Mislabeled Days')

    sns.lineplot(data=combined_data[combined_data['source'] == 'Test'], 
                 x=combined_data[combined_data['source'] == 'Test'].index, 
                 y=feature, color=colors[1], marker='o', ax=axes[i], label='Test')
    
    axes[i].set_title(f'{feature} Over Time')
    axes[i].set_xlabel('ID')
    axes[i].set_ylabel(feature)
    axes[i].grid(color='gray', linestyle='--', linewidth=0.7)
    axes[i].legend()

plt.tight_layout()
plt.show()



cmap = plt.get_cmap('BrBG')
colors = [cmap(0.8), cmap(0.2), cmap(0)]

# List of numerical features (excluding 'day' and 'rainfall')
numerical_features = ['pressure', 'maxtemp', 'temperature', 'mintemp', 
                      'dewpoint', 'humidity', 'cloud', 'sunshine', 
                      'winddirection', 'windspeed']

fig, axes = plt.subplots(len(numerical_features), 1, figsize=(15, len(numerical_features) * 3))

for i, feature in enumerate(numerical_features):
    rolling_max = train_data[feature].rolling(window=7).max()
    rolling_mean = train_data[feature].rolling(window=7).mean()
    rolling_min = train_data[feature].rolling(window=7).min()
    
    axes[i].plot(rolling_max, label='Max', color=colors[0])
    axes[i].plot(rolling_mean, label='Mean', color=colors[1])
    axes[i].plot(rolling_min, label='Min', color=colors[2])
    
    mislabeled_indices = train_data[train_data['day_mislabelled']].index
    mislabeled_values = train_data.loc[mislabeled_indices, feature]
    axes[i].scatter(mislabeled_indices, mislabeled_values, color='red', marker='x', zorder=3, label='Mislabeled Days')
    
    axes[i].set_title(f'{feature} Over Time')
    axes[i].set_xlabel('Index')
    axes[i].set_ylabel(feature)
    axes[i].grid(color='gray', linestyle='--', linewidth=0.7)
    axes[i].legend()

plt.tight_layout()
plt.show()



# Impute mislabeled days with expected values
train_data.loc[train_data['day_mislabelled'], 'day'] = train_data.loc[train_data['day_mislabelled'], 'expected_day']

# Verify imputation
print("\nImputed Days:")
for idx, row in train_data[train_data['day_mislabelled']].iterrows():
    print(f"Index: {idx}, Expected Day: {row['expected_day']}, Actual Day (Imputed): {row['day']}")



# Visualize expected vs actual days after imputation
plt.figure(figsize=(15, 5))
plt.plot(train_data.index, train_data['expected_day'], label='Expected Day', marker='s', linestyle='-', color=colors[0])
plt.plot(train_data.index, train_data['day'], label='Actual Day (After Imputation)', marker='o', linestyle='-', color=colors[1])

# Highlight previously mislabeled days
mislabeled_indices = train_data[train_data['day_mislabelled']].index
plt.scatter(mislabeled_indices, train_data.loc[mislabeled_indices, 'day'], color='red', label='Mislabeled Days', zorder=3)

plt.title('Expected vs. Actual Days After Imputation')
plt.xlabel('Index')
plt.ylabel('Day')
plt.legend()
plt.grid(color='gray', linestyle='--', linewidth=0.7)
plt.show()



# Handle missing value in test_data

# Check for missing values before filling
print("Missing Values Before Filling:")
print(test_data['winddirection'].isnull().sum())

# Fill missing value with mean
test_data['winddirection'].fillna(test_data['winddirection'].mean(), inplace=True)

# Check for missing values after filling
print("\nMissing Values After Filling:")
print(test_data['winddirection'].isnull().sum())



train_data = train_data.drop(['source', 'expected_day', 'day_mislabelled'], axis=1)
test_data = test_data.drop('source', axis=1)


# Select numerical features for train and original datasets
train_numerical_features = train_data.select_dtypes(include=['int64', 'float64']).columns.tolist()
original_numerical_features = original_data.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Select numerical features for test dataset
test_numerical_features = test_data.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Create heatmaps in one column and three rows
fig, axes = plt.subplots(3, 1, figsize=(10, 18))  
# Train Data Heatmap
sns.heatmap(train_data[train_numerical_features].corr(), cmap='BrBG', annot=True, fmt='.2f', ax=axes[0])
axes[0].set_title('Correlation Heatmap - Train Data')
# Original Data Heatmap
sns.heatmap(original_data[original_numerical_features].corr(), cmap='BrBG', annot=True, fmt='.2f', ax=axes[1])
axes[1].set_title('Correlation Heatmap - Original Data')
# Test Data Heatmap
sns.heatmap(test_data[test_numerical_features].corr(), cmap='BrBG', annot=True, fmt='.2f', ax=axes[2])
axes[2].set_title('Correlation Heatmap - Test Data')

plt.tight_layout()
plt.show()



# Set Training Dataset
X = train_data.drop(['rainfall'], axis=1)
y = train_data['rainfall']


# Set Test Dataset
test_X = test_data

# Apply StandardScaler
#scaler = MinMaxScaler()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
test_X_scaled = scaler.transform(test_X)


# Define models
models = {
    'tabpfn': TabPFNClassifier(
        random_state=0, 
        device='cuda'   
    ),  
    'xgboost': XGBClassifier(
        n_jobs=4,
        random_state=0,
        tree_method='gpu_hist'  
    ),
    'lightgbm': LGBMClassifier(
        n_jobs=4,
        random_state=0,
        device='gpu',  
        verbose=-1     
    ),
    'catboost': CatBoostClassifier(
        thread_count=4,
        random_state=0,
        task_type='GPU',  
        verbose=0         
    )
}


# Cross-validation setup
kfold = StratifiedKFold(10, shuffle=True, random_state=0)

# Initialize lists to store results
auc_scores = {model_name: [] for model_name in models.keys()}
fpr_dict = {model_name: [] for model_name in models.keys()}
tpr_dict = {model_name: [] for model_name in models.keys()}

# Perform cross-validation
for fold, (train_idx, val_idx) in enumerate(kfold.split(X_scaled, y)):
    X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    print(f"\nFold {fold+1}:")
    
    # Train models
    for model_name, model in models.items():
        model.fit(X_train, y_train)

        # Predict probabilities
        y_pred_proba = model.predict_proba(X_val)[:, 1]

        # Calculate AUC
        auc = roc_auc_score(y_val, y_pred_proba)
        auc_scores[model_name].append(auc)
        
        print(f"Model: {model_name}, AUC: {auc}")

        # Calculate fpr and tpr for plotting
        fpr, tpr, _ = roc_curve(y_val, y_pred_proba)
        fpr_dict[model_name].append(fpr)
        tpr_dict[model_name].append(tpr)

# Calculate average AUC for each model
for model_name, scores in auc_scores.items():
    avg_auc = np.mean(scores)
    print(f"\nModel: {model_name}, Average AUC: {avg_auc}")


cmap = plt.get_cmap('BrBG')
colors = [cmap(0.1), cmap(0.3), cmap(0.7), cmap(0.9)]

plt.figure(figsize=(10, 8))
for i, model_name in enumerate(models.keys()):
    mean_fpr = np.linspace(0, 1, 100)
    mean_tpr = np.zeros_like(mean_fpr)
    for fpr, tpr in zip(fpr_dict[model_name], tpr_dict[model_name]):
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        mean_tpr += interp_tpr
    mean_tpr /= len(fpr_dict[model_name])
    plt.plot(mean_fpr, mean_tpr, label=f"{model_name} (AUC = {np.mean(auc_scores[model_name]):.2f})", color=colors[i])

plt.plot([0, 1], [0, 1], linestyle="--", color="r", label="Chance")
plt.title('ROC Curves for K-Fold Cross Validation')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.grid(color='gray', linestyle='--', linewidth=0.7)
plt.show()



best_model_name = max(auc_scores, key=lambda x: np.mean(auc_scores[x]))
print(f"Best Model: {best_model_name}")
best_model = models[best_model_name]


best_model.fit(X_scaled, y)


test_pred_proba = best_model.predict_proba(test_X_scaled)[:, 1]


submission_df = pd.DataFrame({'id': test_data.index, 'rainfall': test_pred_proba})
submission_df.to_csv('submission.csv', index=False)
print("Submission file created successfully!")


plt.figure(figsize=(12, 4))
sns.histplot(test_pred_proba, kde=True, bins=20, color=colors[3])
plt.title('Distribution of Predicted Probabilities on Test Set')
plt.xlabel('Predicted Probability')
plt.ylabel('Frequency')
plt.grid(color='gray', linestyle='--', linewidth=0.7)
plt.show()



submission_df.head(10)

