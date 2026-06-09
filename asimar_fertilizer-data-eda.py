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



import pandas as pd
test_data = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')
#test_data.head()
train_data = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
train_data.head()


train_data.info()


summary= train_data.describe()  ##defining a summary of the data 
count_soil = train_data['Soil Type'].value_counts()
frequent_soil = count_soil.idxmax()
#frequency_soil =count_soil.max()
count_crop = train_data['Crop Type'].value_counts()
frequent_crop = count_crop.idxmax()
#frequency_crop = count_crop.max()
count_fert = train_data['Fertilizer Name'].value_counts()
frequent_fert = count_fert.idxmax()
#frequency_fert = count_fert.max()
#print(count_soil, count_crop,count_fert)
print("Frequent Soil:", frequent_soil,
      "\nFrequent Crop:",frequent_crop,
      "\nFrequent Fertiliser:",frequent_fert)
train_data[['Temparature','Humidity','Moisture','Nitrogen','Potassium','Phosphorous']].describe().T


#get counts
count_crop = train_data['Crop Type'].value_counts()

import matplotlib.pyplot as plt
# Plot pie chart
count_crop.plot.pie(autopct='%1.1f%%', startangle=90, figsize=(5, 5), colors=['skyblue', 'orange', 'lightgreen'])
plt.title('Distribution of Crop Types')
plt.ylabel('')  # Remove y-label
plt.show()


#get counts
count_soil = train_data['Soil Type'].value_counts()

# Plot pie chart
count_soil.plot.pie(autopct='%1.1f%%', startangle=90, figsize=(5, 5), colors=['skyblue', 'orange', 'lightgreen'])
plt.title('Distribution of soil Types')
plt.ylabel('')  # Remove y-label
plt.show()


import seaborn as sns
corr = train_data.corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()


#temp vs crop type

# Colorful boxplot with a custom palette
sns.boxplot(data=train_data, x='Crop Type', y='Temparature', palette='pastel')

# Add title with better formatting
plt.title('Crop Type vs Temperature', fontsize=14, fontweight='bold', color='darkblue')

# Improve x-axis labels for readability if needed
plt.xticks(rotation=45, fontsize=10)
plt.yticks(fontsize=10)

# Add gridlines for better readability
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Show plot
plt.tight_layout()
plt.show()


#temp vs soil type

# Colorful boxplot with a custom palette
sns.boxplot(data=train_data, x='Soil Type', y='Temparature', palette='Set2')

# Add title with better formatting
plt.title('Soil Type vs Temperature', fontsize=14, fontweight='bold', color='darkblue')

# Improve x-axis labels for readability if needed
plt.xticks(rotation=45, fontsize=10)
plt.yticks(fontsize=10)

# Add gridlines for better readability
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Show plot
plt.tight_layout()
plt.show()


# Colorful boxplot with a custom palette
sns.boxplot(data=train_data, y='Humidity', x='Fertilizer Name', palette='Set2')

# Add title with better formatting
plt.title('Humidity vs Fertilizer Name', fontsize=14, fontweight='bold', color='darkblue')

# Improve x-axis labels for readability if needed
plt.xticks( fontsize=10)
plt.yticks(fontsize=10)

# Add gridlines for better readability
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Show plot
plt.tight_layout()
plt.show()

# Colorful boxplot with a custom palette
sns.boxplot(data=train_data, y='Humidity', x='Crop Type', palette='Set2')

# Add title with better formatting
plt.title('Humidity vs Crop type', fontsize=14, fontweight='bold', color='darkblue')

# Improve x-axis labels for readability if needed
plt.xticks(rotation=45, fontsize=10)
plt.yticks(fontsize=10)

# Add gridlines for better readability
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Show plot
plt.tight_layout()
plt.show()

# Colorful boxplot with a custom palette
sns.boxplot(data=train_data, y='Humidity', x='Soil Type', palette='Set2')

# Add title with better formatting
plt.title('Humidity vs Soil type', fontsize=14, fontweight='bold', color='darkblue')

# Improve x-axis labels for readability if needed
plt.xticks( fontsize=10)
plt.yticks(fontsize=10)

# Add gridlines for better readability
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Show plot
plt.tight_layout()
plt.show()


# Step 1: Group by Crop Type and compute average N, P, K
avg_npk = train_data.groupby('Crop Type')[['Nitrogen', 'Phosphorous', 'Potassium']].mean()

# Step 2: Plot heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(avg_npk, annot=True, fmt=".1f", cmap="coolwarm", cbar_kws={'label': 'Average Value'})

# Step 3: Customize plot
plt.title('Average NPK Levels by Crop Type', fontsize=14, fontweight='bold')
plt.xlabel('Nutrient Type')
plt.ylabel('Crop Type')
plt.xticks(rotation=0)
plt.yticks(rotation=0)

plt.tight_layout()
plt.show()


# Count plot: Fertilizer distribution by Crop Type
plt.figure(figsize=(10, 6))
sns.countplot(data=train_data, x='Crop Type', hue='Fertilizer Name', palette='Set2')

plt.title('Fertilizer Usage by Crop Type', fontsize=14, fontweight='bold')
plt.xlabel('Crop Type')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.legend(title='Fertilizer')
plt.tight_layout()
plt.show()


# Create a frequency table (contingency table)
cf_table = pd.crosstab(train_data['Crop Type'], train_data['Fertilizer Name'])

# Plot heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(cf_table, annot=True, fmt='d', cmap='YlGnBu')

plt.title('Heatmap of Crop Type vs Fertilizer Name', fontsize=14, fontweight='bold')
plt.xlabel('Fertilizer Name')
plt.ylabel('Crop Type')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()


top_n = 3

# Step 1: Get top N Crop Types
top_crops = train_data['Crop Type'].value_counts().nlargest(top_n).index

# Step 2: Get top N Fertiliser Names
top_ferts = train_data['Fertilizer Name'].value_counts().nlargest(top_n).index

# Step 3: Filter the DataFrame
filtered_data = train_data[train_data['Crop Type'].isin(top_crops) & 
                           train_data['Fertilizer Name'].isin(top_ferts)]

# Step 4: Create crosstab
cf_table = pd.crosstab(filtered_data['Crop Type'], filtered_data['Fertilizer Name'])

# Step 5: Plot heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cf_table, annot=True, fmt='d', cmap='coolwarm')

plt.title(f'Heatmap of Top {top_n} Crop Types vs Top {top_n} Fertilizers', fontsize=14)
plt.xlabel('Fertilizer Name')
plt.ylabel('Crop Type')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()


# Step 3: Create crosstab (Crop Type as index)
cf_table = pd.crosstab(train_data['Crop Type'], train_data['Fertilizer Name'])

# Step 4: Plot stacked bar chart
cf_table.plot(kind='bar', stacked=True, figsize=(10, 6), colormap='tab20')

# Step 5: Customize
plt.title(f'Stacked Bar Chart: Crops vs Fertilizer Usage', fontsize=14)
plt.xlabel('Crop Type')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.legend(title='Fertilizer Name', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# Count plot: Fertilizer distribution by Crop Type
plt.figure(figsize=(10, 6))
sns.countplot(data=train_data, x='Crop Type', hue='Fertilizer Name', palette='Set2')

plt.title('Fertilizer Usage by Crop Type', fontsize=14, fontweight='bold')
plt.xlabel('Crop Type')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.legend(title='Fertilizer')
plt.tight_layout()
plt.show()


top_n = 5

# Step 2: Filter for top N crops and fertilisers
top_crops = train_data['Crop Type'].value_counts().nlargest(top_n).index
top_ferts = train_data['Fertilizer Name'].value_counts().nlargest(top_n).index

filtered_data = train_data[
    train_data['Crop Type'].isin(top_crops) & 
    train_data['Fertilizer Name'].isin(top_ferts)
]

# Step 3: Create crosstab (Crop Type as index)
cf_table = pd.crosstab(filtered_data['Crop Type'], filtered_data['Fertilizer Name'])

# Step 4: Plot stacked bar chart
cf_table.plot(kind='bar', stacked=True, figsize=(10, 6), colormap='tab20')

# Step 5: Customize
plt.title(f'Stacked Bar Chart: Top {top_n} Crops vs Fertilizer Usage', fontsize=14)
plt.xlabel('Crop Type')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.legend(title='Fertilizer Name', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


fs_table = pd.crosstab(train_data['Fertilizer Name'], train_data['Soil Type'])
fs_table.plot(kind='bar', stacked=True, figsize=(10, 6), colormap='Set3')

plt.title('Fertilizer Usage by Soil Type (Stacked Bar)', fontsize=14)
plt.xlabel('Fertilizer Name')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.legend(title='Soil Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

sns.countplot(data=train_data, x='Fertilizer Name', hue='Soil Type', palette='pastel')
plt.title('Fertilizer Name vs Soil Type (Grouped Bar)', fontsize=14)
plt.xlabel('Fertilizer Name')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.legend(title='Soil Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


# Plot heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(fs_table, annot=True, fmt='d', cmap='YlOrBr')

plt.title('Fertilizer Usage by Soil Type (Heatmap)', fontsize=14)
plt.xlabel('Soil Type')
plt.ylabel('Fertilizer Name')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()


import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(18, 5))

# Nitrogen vs Soil Type
plt.subplot(1, 3, 1)
sns.boxplot(data=train_data, x='Soil Type', y='Nitrogen', palette='Set2')
plt.title('Nitrogen vs Soil Type')
plt.xticks(rotation=45)

# Phosphorous vs Soil Type
plt.subplot(1, 3, 2)
sns.boxplot(data=train_data, x='Soil Type', y='Phosphorous', palette='Set2')
plt.title('Phosphorous vs Soil Type')
plt.xticks(rotation=45)

# Potassium vs Soil Type
plt.subplot(1, 3, 3)
sns.boxplot(data=train_data, x='Soil Type', y='Potassium', palette='Set2')
plt.title('Potassium vs Soil Type')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()


import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

# Step 2: Select input features and target
features = ['Temparature', 'Humidity', 'Moisture', 'Soil Type', 'Crop Type',
            'Nitrogen', 'Potassium', 'Phosphorous']
target = 'Fertilizer Name'

X = train_data[features]
y = train_data[target]

# Step 3: Encode categorical variables
le_soil = LabelEncoder()
le_crop = LabelEncoder()
le_target = LabelEncoder()

X['Soil Type'] = le_soil.fit_transform(X['Soil Type'])
X['Crop Type'] = le_crop.fit_transform(X['Crop Type'])
y_encoded = le_target.fit_transform(y)

# Step 4: Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Step 5: Train Random Forest (This part is not strictly needed before grid search, but it was in the original cell)
rf = RandomForestClassifier(n_estimators=100,random_state=42)
rf.fit(X_train, y_train)

# Step 6: Evaluate (This part is not strictly needed before grid search, but it was in the original cell)
y_pred = rf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=le_target.classes_))


from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Define a smaller parameter grid
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [None, 10],
    'min_samples_split': [2, 5],
}

# Initialize model
rf_base = RandomForestClassifier(random_state=42)

# Grid search with 5-fold cross-validation
grid_search = GridSearchCV(estimator=rf_base, param_grid=param_grid, 
                           cv=5, n_jobs=-1, scoring='accuracy', verbose=1)

# Fit to training data
grid_search.fit(X_train, y_train)

# Best parameters
print("Best Parameters:", grid_search.best_params_)

# Evaluate best estimator on test set
best_rf = grid_search.best_estimator_
y_pred = best_rf.predict(X_test)

print("Accuracy on Test Set:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=le_target.classes_))



import pandas as pd
import numpy as np
import logging
import time
from cuml.ensemble import RandomForestClassifier as cuRFClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

test_data = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')
#test_data.head()
train_data = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
train_data.head()

features = ['Temparature', 'Humidity', 'Moisture', 'Soil Type', 'Crop Type',
            'Nitrogen', 'Potassium', 'Phosphorous']
target = 'Fertilizer Name'

X = train_data[features]
y = train_data[target]
# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def optimize_memory(df):
    """Optimize memory usage of dataframe"""
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype('category')
        elif df[col].dtype == 'float64':
            df[col] = df[col].astype('float32')
        elif df[col].dtype == 'int64':
            df[col] = df[col].astype('int32')
    return df

# Step 1: Load and optimize data
logger.info("Loading and optimizing data...")
start_time = time.time()

# Assuming train_data is already loaded


X = train_data[features].copy()
y = train_data[target].copy()

# Optimize memory usage
X = optimize_memory(X)
logger.info(f"Memory optimization completed in {time.time() - start_time:.2f} seconds")

# Step 2: Fast encoding of categorical variables
logger.info("Encoding categorical variables...")
encode_start = time.time()

# Convert categorical columns to codes directly (faster than LabelEncoder)
X['Soil Type'] = X['Soil Type'].cat.codes
X['Crop Type'] = X['Crop Type'].cat.codes

# For target variable, we still need LabelEncoder for inverse transform later
le_target = LabelEncoder()
y_encoded = le_target.fit_transform(y)

logger.info(f"Encoding completed in {time.time() - encode_start:.2f} seconds")

# Step 3: Split data
logger.info("Splitting data into train/test sets...")
split_start = time.time()

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

logger.info(f"Data split completed in {time.time() - split_start:.2f} seconds")

# Step 4: Train GPU Random Forest
logger.info("Training GPU Random Forest model...")
train_start = time.time()

# Initialize and train GPU Random Forest
rf = cuRFClassifier(
    n_estimators=750,
    random_state=42,
    n_streams=1  # Utilize both T4 GPUs
)
rf.fit(X_train, y_train)

logger.info(f"Model training completed in {time.time() - train_start:.2f} seconds")

# Step 5: Evaluate
logger.info("Evaluating model...")
eval_start = time.time()

y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
logger.info(f"Model accuracy: {accuracy:.4f}")
logger.info("\nClassification Report:")
logger.info(classification_report(y_test, y_pred, target_names=le_target.classes_))

logger.info(f"Evaluation completed in {time.time() - eval_start:.2f} seconds")
logger.info(f"Total execution time: {time.time() - start_time:.2f} seconds")


import pandas as pd
import numpy as np
import logging
import time
from cuml.ensemble import RandomForestClassifier as cuRFClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

test_data = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')
#test_data.head()
train_data = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
print(train_data.head())

# Set up logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def print_and_log(message, level='info'):
    """Helper function to both print and log messages"""
    print(f"\n{'='*80}\n{message}\n{'='*80}")
    if level == 'info':
        logger.info(message)
    elif level == 'warning':
        logger.warning(message)
    elif level == 'error':
        logger.error(message)

def optimize_memory(df):
    """Optimize memory usage of dataframe"""
    print_and_log("Starting memory optimization...")
    initial_memory = df.memory_usage(deep=True).sum() / 1024**2  # in MB
    
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype('category')
        elif df[col].dtype == 'float64':
            df[col] = df[col].astype('float32')
        elif df[col].dtype == 'int64':
            df[col] = df[col].astype('int32')
    
    final_memory = df.memory_usage(deep=True).sum() / 1024**2  # in MB
    memory_saved = initial_memory - final_memory
    print_and_log(f"Memory optimization complete. Saved {memory_saved:.2f} MB")
    return df

# Step 1: Load and optimize data
print_and_log("Starting data processing pipeline...")
start_time = time.time()

# Assuming train_data is already loaded
features = ['Temparature', 'Humidity', 'Moisture', 'Soil Type', 'Crop Type',
            'Nitrogen', 'Potassium', 'Phosphorous']
target = 'Fertilizer Name'

print_and_log(f"Selected features: {', '.join(features)}")
print_and_log(f"Target variable: {target}")

X = train_data[features].copy()
y = train_data[target].copy()

print_and_log(f"Initial data shape: {X.shape}")
print_and_log(f"Number of unique target classes: {y.nunique()}")

# Optimize memory usage
X = optimize_memory(X)
logger.info(f"Memory optimization completed in {time.time() - start_time:.2f} seconds")

# Step 2: Fast encoding of categorical variables
print_and_log("Starting categorical variable encoding...")
encode_start = time.time()

# Convert categorical columns to codes directly (faster than LabelEncoder)
X['Soil Type'] = X['Soil Type'].cat.codes
X['Crop Type'] = X['Crop Type'].cat.codes

# For target variable, we still need LabelEncoder for inverse transform later
le_target = LabelEncoder()
y_encoded = le_target.fit_transform(y)

print_and_log(f"Encoding completed in {time.time() - encode_start:.2f} seconds")
print_and_log(f"Encoded target classes: {le_target.classes_}")

# Step 3: Split data
print_and_log("Splitting data into train/test sets...")
split_start = time.time()

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

print_and_log(f"Train set shape: {X_train.shape}")
print_and_log(f"Test set shape: {X_test.shape}")
logger.info(f"Data split completed in {time.time() - split_start:.2f} seconds")

# Step 4: Train GPU Random Forest
print_and_log("Initializing GPU Random Forest model...")
print_and_log("Model parameters:")
print(f"- Number of estimators: rf.n_estimators")
print(f"- Number of GPUs: 2")
print(f"- Random state: 42")

train_start = time.time()

# Initialize and train GPU Random Forest
rf = cuRFClassifier(
    n_estimators= 1500,
    random_state=42,
    n_streams=1  # Utilize both T4 GPUs
)

print_and_log("Starting model training...")
rf.fit(X_train, y_train)

training_time = time.time() - train_start
print_and_log(f"Model training completed in {training_time:.2f} seconds")
print_and_log(f"Average time per tree: {training_time/rf.n_estimators:.4f} seconds")

# Step 5: Evaluate
print_and_log("Starting model evaluation...")
eval_start = time.time()

y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print_and_log("Model Performance Metrics:")
print(f"Accuracy: {accuracy:.4f}")
print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=le_target.classes_))

logger.info(f"Evaluation completed in {time.time() - eval_start:.2f} seconds")

# Final summary
total_time = time.time() - start_time
print_and_log("Pipeline Execution Summary:")
print(f"Total execution time: {total_time:.2f} seconds")
print(f"Memory optimization time: {encode_start - start_time:.2f} seconds")
print(f"Encoding time: {split_start - encode_start:.2f} seconds")
print(f"Training time: {training_time:.2f} seconds")
print(f"Evaluation time: {time.time() - eval_start:.2f} seconds")


rf.n_estimators

