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

# File paths
train_path = r"/kaggle/input/playground-series-s5e7/train.csv"
test_path = r"/kaggle/input/playground-series-s5e7/test.csv"

# Read the datasets
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
# Drop 'id' column if it exists
train_df = train_df.drop(columns=['id'], errors='ignore')
test_df = test_df.drop(columns=['id'], errors='ignore')


# Display the first few rows to check the data
print("Train Set:")
train_df.head()


# Display the first few rows to check the data
print("Test Set:")
test_df.head()


import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=FutureWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    # your code here (e.g., imputation or plotting that triggers warnings)



def identify_variable_types(df):
    """
    Identifies numerical and categorical columns in a DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to analyze.

    Returns:
        tuple: (list of numerical columns, list of categorical columns)
    """
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    return num_cols, cat_cols



# For train set
train_num_cols, train_cat_cols = identify_variable_types(train_df)
print("Train - Numerical Columns:", train_num_cols)
print("Train - Categorical Columns:", train_cat_cols)

# For test set
test_num_cols, test_cat_cols = identify_variable_types(test_df)
print("\nTest - Numerical Columns:", test_num_cols)
print("Test - Categorical Columns:", test_cat_cols)



def missing_values_by_type(df):
    """
    Returns the count of missing values in numerical and categorical columns separately.

    Args:
        df (pd.DataFrame): The DataFrame to analyze.

    Returns:
        tuple: Two DataFrames showing missing values in numerical and categorical columns.
    """
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    
    num_missing = df[num_cols].isnull().sum()
    cat_missing = df[cat_cols].isnull().sum()
    
    num_missing = num_missing[num_missing > 0].sort_values(ascending=False)
    cat_missing = cat_missing[cat_missing > 0].sort_values(ascending=False)
    
    return num_missing, cat_missing



# Train set
train_num_missing, train_cat_missing = missing_values_by_type(train_df)
print("Train - Missing Values in Numerical Columns:\n", train_num_missing)
print("\nTrain - Missing Values in Categorical Columns:\n", train_cat_missing)

# Test set
test_num_missing, test_cat_missing = missing_values_by_type(test_df)
print("\nTest - Missing Values in Numerical Columns:\n", test_num_missing)
print("\nTest - Missing Values in Categorical Columns:\n", test_cat_missing)



import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno

def visualize_missing_values(df, title='Missing Values Overview'):
    """
    Plots charts to visualize missing values:
    - Bar plot of missing % by column
    - Heatmap of missing patterns
    - Optional: Matrix plot using missingno
    
    Args:
        df (pd.DataFrame): The DataFrame to visualize.
        title (str): Title prefix for the plots.
    """
    # --- Missing % per column ---
    missing_percent = df.isnull().mean() * 100
    missing_percent = missing_percent[missing_percent > 0].sort_values(ascending=False)

    if not missing_percent.empty:
        plt.figure(figsize=(10, 5))
        sns.barplot(x=missing_percent.values, y=missing_percent.index, palette='viridis')
        plt.title(f'{title}: Missing Value Percentages')
        plt.xlabel('Missing Percentage (%)')
        plt.ylabel('Columns')
        plt.grid(axis='x', linestyle='--', alpha=0.6)
        for i, v in enumerate(missing_percent.values):
            plt.text(v + 0.5, i, f'{v:.1f}%', va='center')
        plt.tight_layout()
        plt.show()
    else:
        print("âœ… No missing values to plot.")

    # --- Missing pattern heatmap ---
    plt.figure(figsize=(12, 6))
    sns.heatmap(df.isnull(), cbar=False, cmap='YlOrRd', yticklabels=False)
    plt.title(f'{title}: Missing Pattern Heatmap')
    plt.xlabel("Columns")
    plt.tight_layout()
    plt.show()

    # --- Missingno matrix plot (optional, shows streaks) ---
    msno.matrix(df, figsize=(12, 6))
    plt.title(f'{title}: Missing Data Matrix')
    plt.show()



visualize_missing_values(train_df, title='Train Set')
visualize_missing_values(test_df, title='Test Set')



def impute_missing_values(train_df, test_df):
    train_df = train_df.copy()
    test_df = test_df.copy()

    # Numerical columns
    num_cols = train_df.select_dtypes(include=['int64', 'float64']).columns
    for col in num_cols:
        mean_value = train_df[col].mean()
        train_df[col] = train_df[col].fillna(mean_value)
        if col in test_df.columns:
            test_df[col] = test_df[col].fillna(mean_value)

    # Categorical columns
    cat_cols = train_df.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        mode_value = train_df[col].mode()[0]
        train_df[col] = train_df[col].fillna(mode_value)
        if col in test_df.columns:
            test_df[col] = test_df[col].fillna(mode_value)

    return train_df, test_df



train_df, test_df = impute_missing_values(train_df, test_df)

# Confirm no missing values remain
print("Train missing values:\n", train_df.isnull().sum().sum())
print("Test missing values:\n", test_df.isnull().sum().sum())



import matplotlib.pyplot as plt
import seaborn as sns

def plot_numerical_columns(df, max_cols=4):
    """
    Plots histogram with KDE and boxplot for each numerical column in the DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing data.
        max_cols (int): Maximum number of plots per row.
    """
    # Identify numerical columns
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    for col in num_cols:
        plt.figure(figsize=(14, 5))

        # Histogram with KDE
        plt.subplot(1, 2, 1)
        sns.histplot(df[col], kde=True, bins=30, color='skyblue')
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Count')

        # Boxplot
        plt.subplot(1, 2, 2)
        sns.boxplot(x=df[col], color='salmon')
        plt.title(f'Boxplot of {col}')
        plt.xlabel(col)

        plt.tight_layout()
        plt.show()



plot_numerical_columns(train_df)



import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def plot_categorical_percentages_grid(df, cols_per_row=3):
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    total_plots = len(cat_cols)
    rows = int(np.ceil(total_plots / cols_per_row))

    fig, axes = plt.subplots(rows, cols_per_row, figsize=(5 * cols_per_row, 4 * rows))
    axes = axes.flatten() if total_plots > 1 else [axes]

    for i, col in enumerate(cat_cols):
        ax = axes[i]
        percent = df[col].value_counts(normalize=True) * 100
        percent_df = percent.reset_index()
        percent_df.columns = [col, 'Percentage']

        sns.barplot(
            data=percent_df,
            x=col,
            y='Percentage',
            palette='pastel',
            ax=ax
        )

        ax.set_title(f"{col} Distribution (%)")
        ax.set_ylabel("Percentage")
        ax.set_xlabel(col)

        # Optional: Show percentages on bars
        for container in ax.containers:
            ax.bar_label(container, fmt="%.1f%%", label_type="edge", padding=3)

    # Remove empty subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()



plot_categorical_percentages_grid(train_df, cols_per_row=3)



import seaborn as sns
import matplotlib.pyplot as plt

def plot_correlation_heatmap(df, title='Correlation Heatmap (Numerical Features)'):
    """
    Plots a correlation heatmap for numerical variables in the DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame.
        title (str): Title for the plot.
    """
    # Select only numerical columns
    num_df = df.select_dtypes(include=['int64', 'float64'])

    # Compute correlation matrix
    corr_matrix = num_df.corr()

    # Plot heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True,
                linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title(title, fontsize=14)
    plt.tight_layout()
    plt.show()



plot_correlation_heatmap(train_df)



from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Step 1: Label encode the target variable in train set
target_encoder = LabelEncoder()
train_df['Personality'] = target_encoder.fit_transform(train_df['Personality'])

# Step 2: Add flag to distinguish train/test before combining
train_df['is_train'] = 1
test_df['is_train'] = 0

# Step 3: Add dummy target to test to enable consistent encoding
test_df['Personality'] = -1  # Placeholder, will be removed later

# Step 4: Combine train and test
combined_df = pd.concat([train_df, test_df], axis=0)

# Step 5: One-hot encode categorical feature columns only
combined_encoded = pd.get_dummies(combined_df, columns=['Stage_fear', 'Drained_after_socializing'], drop_first=True)

# Step 6: Separate back into train and test
train_encoded = combined_encoded[combined_encoded['is_train'] == 1].drop(['is_train'], axis=1)
test_encoded = combined_encoded[combined_encoded['is_train'] == 0].drop(['is_train', 'Personality'], axis=1)

# Final check
print("âœ… Encoded Train Shape:", train_encoded.shape)
print("âœ… Encoded Test Shape:", test_encoded.shape)



label_map = dict(zip(target_encoder.classes_, target_encoder.transform(target_encoder.classes_)))
print(label_map)



from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import numpy as np
import warnings
warnings.filterwarnings("ignore")

# Prepare features and target
X = train_encoded.drop(columns=['Personality'])
y = train_encoded['Personality']

# Initialize 10-fold stratified CV
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Define models
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "SVM (RBF)": SVC(kernel='rbf', probability=True),
    "MLP": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42),
    "LightGBM": LGBMClassifier(device_type='cpu', random_state=42, verbose=-1),
    "XGBoost": XGBClassifier(tree_method='hist', use_label_encoder=False, eval_metric='logloss', random_state=42),
    "CatBoost": CatBoostClassifier(verbose=0, task_type="CPU", random_seed=42)
}

# Add to your previous cross-validation loop
results = {}

for model_name, model in models.items():
    train_scores = []
    val_scores = []
    confusion_matrices = []
    class_reports = []

    print(f"\nðŸš€ Model: {model_name}")
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(X_train, y_train)

        y_val_pred = model.predict(X_val)
        y_train_pred = model.predict(X_train)

        # Accuracy
        val_acc = accuracy_score(y_val, y_val_pred)
        train_acc = accuracy_score(y_train, y_train_pred)

        # Store metrics
        train_scores.append(train_acc)
        val_scores.append(val_acc)

        # Store confusion matrix and classification report
        confusion_matrices.append(confusion_matrix(y_val, y_val_pred))
        class_reports.append(classification_report(y_val, y_val_pred, output_dict=True))

        print(f"  Fold {fold+1}: Train Acc = {train_acc:.4f}, Val Acc = {val_acc:.4f}")

    results[model_name] = {
        "Train Acc": train_scores,
        "Val Acc": val_scores,
        "Confusion Matrices": confusion_matrices,
        "Classification Reports": class_reports,
        "Model Object": model
    }



def plot_val_accuracy(results_dict):
    plt.figure(figsize=(10, 6))
    for model_name, data in results_dict.items():
        plt.plot(range(1, 11), data["Val Acc"], label=model_name)
    plt.xlabel("Fold")
    plt.ylabel("Validation Accuracy")
    plt.title("Validation Accuracy Across Folds")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()



plot_val_accuracy(results)



import os
import numpy as np
import pandas as pd

# Reuse your existing encoder
# target_encoder is already fitted on train_df['Personality']

# Generate ID column
start_id = 18524
num_test_samples = test_encoded.shape[0]
id_column = list(range(start_id, start_id + num_test_samples))

# Output directory
output_dir = "model_predictions"
os.makedirs(output_dir, exist_ok=True)

# Loop through each model in results
for model_name, data in results.items():
    model = data["Model Object"]
    
    # Get predictions
    y_test_pred = model.predict(test_encoded)

    # âœ… Force convert all integer-type predictions to original labels
    try:
        y_test_labels = target_encoder.inverse_transform(np.array(y_test_pred).astype(int))
    except:
        y_test_labels = y_test_pred  # Use as-is if already string

    # Save predictions to CSV
    submission_df = pd.DataFrame({
        "id": id_column,
        "Personality": y_test_labels
    })


