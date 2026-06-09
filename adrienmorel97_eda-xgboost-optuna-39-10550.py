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


# Basic Libraries
import numpy as np
import pandas as pd
import time
import warnings

# Machine Learning Libraries
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.cluster import KMeans
from sklearn.ensemble import StackingRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Visualization Libraries
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines

# ML Models
from xgboost import XGBRegressor
import lightgbm as lgb

# Target Encoding
from cuml.preprocessing import TargetEncoder

# Hyperparameter Tuning
import optuna

# Progress Bar
from tqdm.auto import tqdm

# Suppress unnecessary warnings
warnings.filterwarnings("ignore", category=UserWarning, module="seaborn")
warnings.filterwarnings("ignore", category=FutureWarning, module="seaborn")
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter('ignore')


train_df=pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
test_df=pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')
train_extra = pd.read_csv("/kaggle/input/playground-series-s5e2/training_extra.csv", index_col='id')
train_df = pd.concat([train_df, train_extra], axis=0, ignore_index=True)


# Check dataset shape and first rows
print(f"Dataset contains {train_df.shape[0]} rows and {train_df.shape[1]} columns.")
train_df.head()


train_df.info()


# Save 'id' column from the test set for submission
test_ids = test_df['id'].copy()

# Drop 'id' column from train and test if present
if 'id' in train_df.columns:
    train_df.drop('id', axis=1, inplace=True)
if 'id' in test_df.columns:
    test_df.drop('id', axis=1, inplace=True)

# Define the target column
target_column = 'Price'

# Identify categorical and numerical columns
categorical_columns = train_df.select_dtypes(include=['object']).columns
numerical_columns = train_df.select_dtypes(exclude=['object']).columns

# Print out column information for verification
print("Target Column:", target_column)
print("\nCategorical Columns:", categorical_columns.tolist())
print("\nNumerical Columns:", numerical_columns.tolist())


train_df.describe().round(2)


for column in categorical_columns:
    num_unique = train_df[column].nunique()
    print(f"'{column}' has {num_unique} unique categories.")


# Print unique value counts for each categorical column
for column in categorical_columns:
    print(f"\nTop value counts in '{column}':\n{train_df[column].value_counts().head(10)}")


# Calculate the percentage of rows containing at least one NaN
nan_rows_percentage = round(train_df.isna().any(axis=1).mean() * 100, 2)
print(f"Percentage of rows with at least one NaN: {nan_rows_percentage}%")


# Calculate missing percentages
missing_percentages = (train_df.isnull().sum() / len(train_df)) * 100

# Convert to a DataFrame for convenience
df_miss = (
    pd.DataFrame({
        "Column": missing_percentages.index,
        "MissingPct": missing_percentages.values
    })
    .sort_values("MissingPct", ascending=False)  # Optional: sort by descending % 
    .reset_index(drop=True)
)

# Create a discrete color palette from magma
colors = sns.color_palette("YlGnBu", len(df_miss))

# Plot
plt.figure(figsize=(8, 4))
sns.barplot(data=df_miss, x="Column", y="MissingPct", palette=colors)
plt.title("Missing Value Percentage by Column")
plt.ylabel("% of Rows Missing")
plt.xticks(rotation=65)
plt.tight_layout()
plt.show()



plt.figure(figsize=(15,9))
plt.title("Visualizing Missing Values")
sns.heatmap(train_df.isnull(), cbar=False, cmap=sns.color_palette('YlGnBu'), yticklabels=False);
plt.show()


# Create a color palette for the columns
palette = sns.color_palette('YlGnBu', len(numerical_columns))
color_dict = dict(zip(numerical_columns, palette))

# Create a grid of subplots for histograms, boxplots, and scatterplots/violin plots
fig = plt.figure(figsize=(25, 8 * len(numerical_columns)))
gs = gridspec.GridSpec(2 * len(numerical_columns), 2, figure=fig)

df_binned = train_df.copy()

for i, column in enumerate(numerical_columns):

    if train_df[column].nunique() > 50: discrete = False
    else : discrete = True
    
    # Plot histogram with a unique color
    ax_hist = fig.add_subplot(gs[2 * i, 0])
    sns.histplot(
        data=train_df, x=column, fill=True, common_norm=False, alpha=0.6,
        linewidth=0.8, color=color_dict[column], ax=ax_hist,  discrete = discrete
    )
    ax_hist.set_title(f'{column} distribution', fontsize=14)

    
    # Plot boxplot with the same unique color
    ax_box = fig.add_subplot(gs[2 * i + 1, 0])
    sns.boxplot(data=train_df, x=column, ax=ax_box, color=color_dict[column])
    sns.despine(ax=ax_box)

    # Conditional plot: violin plot or barplot based on unique values, fallback to scatterplot
    ax_conditional = fig.add_subplot(gs[2 * i:2 * i + 2, 1])  # Merges 2 rows
    if train_df[column].nunique() <= 10:
        # If the column has 10 or fewer unique values, use a violin plot
        sns.violinplot(data=train_df, x=column, y=target_column, ax=ax_conditional, color=color_dict[column], alpha=0.6)
        ax_conditional.set_title(f'{column} vs {target_column} (Violin Plot)', fontsize=14)
    else:
        # Bin the column into 10 intervals, but keep original target column values
        df_binned['Binned Column'] = pd.cut(train_df[column], bins=10)
        sns.violinplot(data=df_binned, x='Binned Column', y=target_column, ax=ax_conditional, color=color_dict[column], alpha=0.6)
        ax_conditional.set_title(f'{column} (Binned) vs {target_column} (Violin Plot)', fontsize=14)
        ax_conditional.set_xlabel(f'{column} (Binned)', fontsize=12)

plt.tight_layout()  # Adjust subplots to fit into the figure area
plt.show()


# Calculate the correlation matrix
correlation_matrix = train_df[numerical_columns].corr()

# Plot the heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="YlGnBu", cbar=True, linewidths=0.5)
plt.title("Correlation Heatmap of Numerical Variables", fontsize=16)
plt.show()



filtered_columns = [col for col in categorical_columns]

fig, axes = plt.subplots(len(filtered_columns), 2, figsize=(15, 5 * len(filtered_columns)))

for i, column in enumerate(filtered_columns):
    # --------------------------------------------------
    # 1) Barplot (LEFT SUBPLOT), sorted by ascending count
    # --------------------------------------------------
    freq_order = (
        train_df[column]
        .value_counts()
        .sort_values(ascending=True)
        .index
    )

    sns.countplot(
        data=train_df,
        x=column,
        order=freq_order,
        ax=axes[i, 0],
        palette='YlGnBu'
    )
    axes[i, 0].set_title(f"Distribution of {column}", fontsize=14)
    axes[i, 0].set_xlabel(column, fontsize=12)
    axes[i, 0].set_ylabel("Count", fontsize=12)
    sns.despine(ax=axes[i, 0])

    # --------------------------------------------------
    # 2) Boxplot (RIGHT SUBPLOT), sorted by ascending mean
    # --------------------------------------------------
    mean_order = (
        train_df.groupby(column)[target_column]
        .mean()
        .sort_values()
        .index
    )

    box_ax = axes[i, 1]
    sns.boxplot(
        data=train_df,
        x=column,
        y=target_column,
        order=mean_order,
        ax=box_ax,
        palette='YlGnBu',
        showmeans=True,
        meanline=True,
        meanprops={
            "color": "red",
            "ls": ":",
            "lw": 2
        }
    )
    box_ax.set_title(f"{column} vs {target_column}", fontsize=14)
    box_ax.set_xlabel(column, fontsize=12)
    box_ax.set_ylabel(target_column, fontsize=12)
    sns.despine(ax=box_ax)

    # Create a custom legend entry for the mean line
    mean_line = mlines.Line2D([], [], color='red', linestyle=':', label='Mean')
    box_ax.legend(handles=[mean_line], loc="upper right")  # Position legend in top-right

plt.tight_layout()
plt.show()



import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ---------------------------------------
# 1) Define the features for looping
# ---------------------------------------
features = ["Material", "Size", "Waterproof", "Style", "Color", "Laptop Compartment"]

# ---------------------------------------
# 2) Sort Brand by Overall Average Price
# ---------------------------------------
brand_order = (
    train_df
    .groupby("Brand")["Price"]
    .mean()
    .sort_values(ascending=True)
    .index
)

# Sort each categorical feature by average price
feature_orders = {}
for feat in features:
    feat_order = (
        train_df
        .groupby(feat)["Price"]
        .mean()
        .sort_values(ascending=True)
        .index
    )
    feature_orders[feat] = feat_order

# ---------------------------------------
# 3) Create Pivot Tables in a Loop
# ---------------------------------------
pivot_tables = {}

for feat in features:
    df_grouped = (
        train_df
        .groupby(["Brand", feat], as_index=False)
        .agg({"Price": "mean"})
    )

    df_pivot = df_grouped.pivot(index="Brand", columns=feat, values="Price")

    # Reorder index and columns
    valid_brands = brand_order.intersection(df_pivot.index)
    df_pivot = df_pivot.loc[valid_brands]

    valid_feats = feature_orders[feat].intersection(df_pivot.columns)
    df_pivot = df_pivot.loc[:, valid_feats]

    pivot_tables[feat] = df_pivot

# ---------------------------------------
# 4) Plot All Heatmaps with Adapted Axes
# ---------------------------------------
n_feats = len(features)
cols = 3  # 3 columns for better spacing
rows = int(np.ceil(n_feats / cols))  # Dynamic row count
fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(6 * cols, 5 * rows))

for i, feat in enumerate(features):
    row, col = divmod(i, cols)  # Compute row & col index dynamically
    ax = axes[row, col] if rows > 1 else axes[col]  # Adjust in case of single row
    sns.heatmap(
        pivot_tables[feat],
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        ax=ax
    )
    
    ax.set_title(f"Avg. Price (Brand vs. {feat})", fontsize=14)
    ax.set_xlabel(feat, fontsize=12)
    
    # Only the first column will have a y-label (Brand names)
    if col == 0:
        ax.set_ylabel("Brand", fontsize=12)
    else:
        ax.set_ylabel("")

# Remove empty subplots if `n_feats` is not a multiple of `cols`
for j in range(i + 1, rows * cols):
    row, col = divmod(j, cols)
    fig.delaxes(axes[row, col])

plt.tight_layout()
plt.show()



# Copy DataFrame to avoid modifying the original
train_df_fe = train_df.copy()


combinations_list = [
    ['Brand', 'Material', 'Size', 'Laptop Compartment', 'Waterproof'],
    ['Brand', 'Material', 'Size', 'Laptop Compartment', 'Style'],
    ['Brand', 'Material', 'Size', 'Laptop Compartment', 'Color'],
    ['Brand', 'Material', 'Size', 'Waterproof', 'Style'],
    ['Brand', 'Material', 'Size', 'Waterproof', 'Color'],
    ['Brand', 'Material', 'Size', 'Style', 'Color'],
    ['Brand', 'Material', 'Laptop Compartment', 'Waterproof', 'Style'],
    ['Brand', 'Material', 'Laptop Compartment', 'Waterproof', 'Color'],
    ['Brand', 'Material', 'Laptop Compartment', 'Style', 'Color'],
    ['Brand', 'Material', 'Waterproof', 'Style', 'Color'],
    ['Brand', 'Size', 'Laptop Compartment', 'Waterproof', 'Style'],
    ['Brand', 'Size', 'Laptop Compartment', 'Waterproof', 'Color'],
    ['Brand', 'Size', 'Laptop Compartment', 'Style', 'Color'],
    ['Brand', 'Size', 'Waterproof', 'Style', 'Color'],
    ['Brand', 'Laptop Compartment', 'Waterproof', 'Style', 'Color'],
    ['Material', 'Size', 'Laptop Compartment', 'Waterproof', 'Style'],
    ['Material', 'Size', 'Laptop Compartment', 'Waterproof', 'Color'],
    ['Material', 'Size', 'Laptop Compartment', 'Style', 'Color'],
    ['Material', 'Size', 'Waterproof', 'Style', 'Color'],
    ['Material', 'Laptop Compartment', 'Waterproof', 'Style', 'Color'],
    ['Size', 'Laptop Compartment', 'Waterproof', 'Style', 'Color']
]


# Function to apply target encoding for a single combination of features
def target_encoding_for_combination(train_df, col_name, target_col):
    """Calculates target encoding for a single combination of features."""
    # Convert list to string for column name and join features
    col_name_str = '_'.join(col_name)  # Convert list of features to a string
    features = col_name
    temp_col = train_df[features].astype(str).agg('_'.join, axis=1)

    # Add the target encoding column (initialized with NaN)
    train_df[f'{col_name_str}_target'] = np.nan

    # Compute mean target value for each category
    mean_map = train_df.groupby(temp_col)[target_col].mean()

    # Map computed means to the dataframe
    train_df[f'{col_name_str}_target'] = temp_col.map(mean_map)

    # Replace NaNs with the global median
    train_df[f'{col_name_str}_target'] = train_df[f'{col_name_str}_target'].fillna(train_df[f'{col_name_str}_target'].median())

    return col_name_str, mean_map

# Function to apply target encoding for all combinations
def apply_target_encoding(train_df, combinations_list, target_col):
    """Applies target encoding directly with the provided combinations."""
    target_encodings = {}

    # Sequentially apply target encoding for each combination in the list
    for col_name in combinations_list:
        col_name_str = '_'.join(col_name)  # Convert list to string for column name
        features = col_name
        temp_col = train_df[features].astype(str).agg('_'.join, axis=1)

        # Add the target encoding column (initialized with NaN)
        train_df[f'{col_name_str}_target'] = np.nan

        # Compute mean target value for each category
        mean_map = train_df.groupby(temp_col)[target_col].mean()

        # Map computed means to the dataframe
        train_df[f'{col_name_str}_target'] = temp_col.map(mean_map)

        # Replace NaNs with the global median
        train_df[f'{col_name_str}_target'] = train_df[f'{col_name_str}_target'].fillna(train_df[f'{col_name_str}_target'].median())

        # Save encoding map for test set
        target_encodings[col_name_str] = mean_map.to_dict()

    print(f"âœ… Target Encoding Completed: {len(target_encodings)} Encoded Features")
    return train_df, target_encodings


def preprocess_dataset(train_df, test_df, target_col):
    """Preprocess the train and test datasets by applying feature engineering steps."""
    
    # -------------------------------
    # Step 1: Imputation for Missing Values
    # -------------------------------
    
    # Identify categorical and numerical features
    categorical_features = train_df.select_dtypes(include=['object']).columns
    numerical_features = ['Weight Capacity (kg)', 'Compartments']  # Assuming these are numerical features

    # Impute missing values for categorical columns with "Missing"
    train_df[categorical_features] = train_df[categorical_features].fillna("Missing")
    test_df[categorical_features] = test_df[categorical_features].fillna("Missing")

    # Impute missing values for numerical columns with median
    imputer = SimpleImputer(strategy='median')
    train_df[numerical_features] = imputer.fit_transform(train_df[numerical_features])
    test_df[numerical_features] = imputer.transform(test_df[numerical_features])
    
    # -------------------------------
    # Step 2: Apply Target Encoding to the Train Dataset
    # -------------------------------
    train_df, target_encodings = apply_target_encoding(train_df, combinations_list, target_col)
    
    # -------------------------------
    # Step 3: Apply Target Encoding to the Test Dataset
    # -------------------------------
    
    def transform_test_df(test_df, target_encodings, combinations_list):
        """Applies the same target encoding transformation to the test dataset."""
        
        # Apply target encoding for each combination
        for col_name in combinations_list:
            col_name_str = '_'.join(col_name)  # Convert the list of features to a string

            if all(f in test_df.columns for f in col_name):  # Check if all features exist in test_df
                temp_col = test_df[col_name].astype(str).agg('_'.join, axis=1)
                test_df[f'{col_name_str}_target'] = temp_col.map(target_encodings[col_name_str])
            else:
                test_df[f'{col_name_str}_target'] = np.nan  # Handle missing combinations
    
            # Replace missing target encodings with the global median from the training set
            test_df[f'{col_name_str}_target'].fillna(train_df[f'{col_name_str}_target'].median(), inplace=True)
    
        return test_df
    
    # Apply target encoding to the test dataset
    test_df = transform_test_df(test_df, target_encodings, combinations_list)

    # -------------------------------
    # Step 4: Standardize Numerical Features
    # -------------------------------

    # Extract target variable and drop it from dataset
    y = train_df[target_col]
    train_df = train_df.drop(columns=[target_col])

    # Define numerical features after feature engineering (excluding target column)
    numerical_features = train_df.select_dtypes(exclude=['object']).columns

    # Initialize and apply StandardScaler for numerical features
    scaler = StandardScaler()
    train_df[numerical_features] = scaler.fit_transform(train_df[numerical_features])
    test_df[numerical_features] = scaler.transform(test_df[numerical_features])

    train_df["Price"] = y  # Reassign target to the train dataset

    return train_df, test_df, y, scaler, target_encodings


# Apply the full preprocessing pipeline
train_df_fe, test_df_fe, y_train, scaler, target_encodings = preprocess_dataset(train_df_fe, test_df, target_column)

print("âœ… Training and testing datasets successfully processed.")


# Get the first combination from target_encodings
first_combination = list(target_encodings.keys())[0]

# Get the encoded values for the first combination
encoding_dict = target_encodings[first_combination]

# Create a dataframe with the interaction and target encoding values
encoding_df = pd.DataFrame(list(encoding_dict.items()), columns=['Interaction', 'Target Encoding'])

# Sort by target encoding value in ascending order
encoding_df = encoding_df.sort_values(by='Target Encoding', ascending=True)

# Plot the scatter plot with index on the x-axis
plt.figure(figsize=(15, 8))
sns.scatterplot(
    data=encoding_df,
    x=range(len(encoding_df)),  # Use the index as x-axis
    y='Target Encoding',
)

# Customize plot
plt.title(f"Target Encoding by {first_combination}", fontsize=16)
plt.xlabel('Number of Points', fontsize=14)  # Label for the x-axis
plt.ylabel('Target Encoding', fontsize=14)

# Display x-ticks only every 100 points
tick_step = 100
ticks = range(0, len(encoding_df), tick_step)
labels = [str(i + 1) for i in ticks]  # Generate labels as numbers from 1 to n
plt.xticks(ticks=ticks, labels=labels, rotation=45, ha="right")  # Show numbers on x-axis

plt.tight_layout()
plt.show()


class XGBPipeline:
    def __init__(self, train: pd.DataFrame, test: pd.DataFrame, test_ids: pd.Series,
                 target: str, features: list, cats: list,
                 te_params: dict = None,
                 sample_frac: float = 0.5,
                 random_state: int = 42):
        """
        Initializes the pipeline with training data, test data, target column, feature columns, categorical features, 
        and target encoding parameters.
        """
        self.train = train.copy()
        self.test = test.copy()
        self.test_ids = test_ids.copy()  # Safely store test IDs
        self.target = target
        self.features = features
        self.cats = cats
        self.sample_frac = sample_frac
        self.random_state = random_state

        # Default target encoder parameters
        if te_params is None:
            te_params = {'n_folds': 25, 'smooth': 20, 'split_method': 'random', 'stat': 'mean'}
        self.te_params = te_params
        self.TE = TargetEncoder(**self.te_params)
    
        self.best_params = None
        self.best_cv_rmse = None
        self.model_xgb = None
        self.model_lgbm = None
        self.model_lr = None
        self.stacking_model = None
        self.best_iteration = None
        self.all_features = self.features

    def hyperparameter_tuning(self, n_trials: int = 20):
        """
        Uses Optuna for hyperparameter tuning of XGBoost on a subsample of the training data.
        This step helps in finding the best hyperparameters for model training.
        """
        print("ğŸ”� Starting hyperparameter tuning with Optuna...")
    
        # Convert categorical columns to 'category' for efficient handling in XGBoost
        self.train[self.cats] = self.train[self.cats].astype('category')
    
        # Check if features and target are correctly defined
        if not self.all_features or self.target not in self.train.columns:
            raise ValueError(f"â�Œ Features or target are incorrectly defined.\n"
                             f"self.all_features: {self.all_features}\n"
                             f"self.target: {self.target}")
    
        # Check for missing features in the training dataset
        missing_features = [col for col in self.all_features if col not in self.train.columns]
        if missing_features:
            raise ValueError(f"â�Œ Missing features in train dataset: {missing_features}")
    
        # Sample the training data for hyperparameter tuning
        train_sample = self.train.sample(frac=self.sample_frac, random_state=self.random_state)
    
        if train_sample.empty or self.target not in train_sample.columns:
            raise ValueError("â�Œ Train sample is empty or target column is missing.")
    
        def objective(trial):
            """
            Defines the objective function for Optuna to optimize the hyperparameters.
            We tune hyperparameters like max_depth, learning_rate, and others for better model performance.
            """
            params = {
                "max_depth": trial.suggest_int("max_depth", 4, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.05, log=True),
                "min_child_weight": trial.suggest_int("min_child_weight", 3, 10),
                "subsample": trial.suggest_float("subsample", 0.9, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.55, 0.65),
                "n_estimators": trial.suggest_int("n_estimators", 600, 1000),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.95, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.35, 0.45)
            }
    
            # Perform 3-fold cross-validation to evaluate the model
            cv = KFold(n_splits=3, shuffle=True, random_state=self.random_state)
            cv_scores = []
            for train_idx, val_idx in cv.split(train_sample):
                if train_sample.empty:
                    raise ValueError("â�Œ Train sample is empty during cross-validation.")
    
                X_train_cv = train_sample.iloc[train_idx][self.all_features]
                y_train_cv = train_sample.iloc[train_idx][self.target]
                X_val_cv = train_sample.iloc[val_idx][self.all_features]
                y_val_cv = train_sample.iloc[val_idx][self.target]
    
                model = XGBRegressor(
                    tree_method="gpu_hist",
                    enable_categorical=True,
                    random_state=self.random_state,
                    **params
                )
                model.fit(
                    X_train_cv, y_train_cv,
                    eval_set=[(X_val_cv, y_val_cv)],
                    eval_metric="rmse",
                    early_stopping_rounds=50,
                    verbose=False
                )
                preds = model.predict(X_val_cv)
                rmse = np.sqrt(mean_squared_error(y_val_cv, preds))
                cv_scores.append(rmse)
            return np.mean(cv_scores)
    
        # Optimize hyperparameters using Optuna
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials)
        self.best_params = study.best_trial.params
        self.best_cv_rmse = study.best_value
        print(f"âœ… Best params: {self.best_params}")
        print(f"âœ… Best CV RMSE: {self.best_cv_rmse}")

    def train_final_model(self):
        """
        Trains the final XGBoost model using the best hyperparameters obtained from Optuna.
        This is done on the full training dataset.
        """
        print("ğŸš€ Training final XGBoost model...")
        self.model_xgb = XGBRegressor(
            tree_method="gpu_hist",
            enable_categorical=True,
            random_state=self.random_state,
            **self.best_params
        )
        self.model_xgb.fit(
            self.train[self.all_features], self.train[self.target],
            eval_metric="rmse",
            verbose=False
        )
        print("âœ… Final model trained.")

    def predict_test(self):
        """
        Generates predictions on the test set using the trained XGBoost model.
        """
        print("ğŸ“Š Generating test predictions...")
    
        # Ensure that categorical columns are encoded properly
        self.test[self.cats] = self.test[self.cats].astype('category')
    
        test_preds = self.model_xgb.predict(self.test[self.all_features])
        return test_preds

    def save_submission(self, predictions, filename="submission.csv"):
        """
        Saves the predictions to a CSV file for submission to Kaggle.
        """
        # Ensure test IDs are in a proper format
        sub = pd.DataFrame({"id": self.test_ids.values, self.target: predictions})
    
        # Ensure filename is a string
        if not isinstance(filename, str):
            raise TypeError("â�Œ `filename` must be a string.")
    
        # Save the submission file
        sub.to_csv(filename, index=False)
        print(f"âœ… Submission saved to {filename}")

    def run_pipeline(self):
        """
        Runs the full pipeline: Preprocessing â†’ Hyperparameter Tuning â†’ Training â†’ Prediction.
        """
        overall_start = time.time()
        
        # Define the steps of the pipeline
        steps = [
            ("Hyperparameter Tuning", self.hyperparameter_tuning),
            ("Training Final Model", self.train_final_model),
        ]
        
        # Run each step sequentially
        with tqdm(total=len(steps), desc="Pipeline Steps", unit="step") as pbar:
            print("ğŸš€ Starting pipeline execution...")
            for step_name, step_func in steps:
                print(f"ğŸ”¹ {step_name}...")
                step_func()  # Execute the function
                pbar.update(1)
                print(f"âœ… Completed: {step_name}")
    
        print("ğŸ“Š Predicting Test Set...")
        predictions = self.predict_test()
    
        print("ğŸ’¾ Saving Submission...")
        self.save_submission(predictions, filename="submission.csv")
    
        overall_elapsed = time.time() - overall_start
        print(f"ğŸ�¯ Pipeline execution complete. Total time: {overall_elapsed:.2f} sec")


# ğŸš€ **Running the pipeline**

# Define necessary variables
target = "Price"
features = [col for col in train_df_fe.columns if col != target]
cats = train_df_fe.select_dtypes(include=['object', 'category']).columns.tolist()

# Initialize the pipeline and run
pipeline = XGBPipeline(train=train_df_fe, test=test_df_fe, test_ids=test_ids, 
                       target=target, features=features, cats=cats)

pipeline.run_pipeline()  # Run the complete pipeline

