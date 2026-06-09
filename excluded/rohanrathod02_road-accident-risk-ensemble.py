# ====================================================
# Setup & Imports
# ====================================================

import os
import warnings
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

import lightgbm as lgb
import xgboost as xgb
import catboost as cb

# Settings
warnings.filterwarnings("ignore")
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)

# Reproducibility
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# ====================================================
# Load Data
# ====================================================

DATA_PATH = "/kaggle/input/playground-series-s5e10"

try:
    train = pd.read_csv(os.path.join(DATA_PATH, "train.csv"))
    test = pd.read_csv(os.path.join(DATA_PATH, "test.csv"))
    sample_submission = pd.read_csv(os.path.join(DATA_PATH, "sample_submission.csv"))

    print("âœ… Data Loaded Successfully")
    print("Train Shape:", train.shape)
    print("Test Shape:", test.shape)
    print("Sample Submission Shape:", sample_submission.shape)
    print()

    # Quick checks
    if "accident_risk" not in train.columns:
        raise ValueError("âš ï¸� Target column 'accident_risk' not found in training data.")

    print("Target Stats (accident_risk):")
    print(train["accident_risk"].describe())
    print()

    # Null check
    print("Missing Values (Train):\n", train.isnull().sum())
    print("Missing Values (Test):\n", test.isnull().sum())

except Exception as e:
    print(f"âš ï¸� Error loading files: {e}")



def data_info(df, df_name):
    """Comprehensive overview of a DataFrame with styled output."""

    print(f"\n{'='*80}")
    print(f"ğŸ“Š Comprehensive Information for DataFrame: {df_name}")
    print(f"{'='*80}\n")

    # --- Shape ---
    print(f"Shape: {df.shape[0]} rows Ã— {df.shape[1]} columns\n")

    # --- Head ---
    print(f"--- {df_name} Head ---\n")
    display(df.head().style.set_table_styles([
        {'selector': 'th', 'props': [('background-color', 'lightblue'), ('color', 'black')]},
        {'selector': 'td', 'props': [('font-size', '10pt')]}
    ], overwrite=False))

    # --- Column Summary ---
    print(f"\n--- {df_name} Column Summary ---\n")
    summary = pd.DataFrame({
        "DataType": df.dtypes,
        "Non-Null Count": df.notnull().sum(),
        "Unique Values": df.nunique(),
        "Missing Values": df.isnull().sum(),
        "Missing %": (df.isnull().sum() / len(df)) * 100
    })
    display(summary.style.set_table_styles([
        {'selector': 'th', 'props': [('background-color', 'lightblue'), ('color', 'black')]},
        {'selector': 'td', 'props': [('font-size', '10pt')]}
    ], overwrite=False))

    # --- Describe (numeric only) ---
    if df.select_dtypes(include=np.number).shape[1] > 0:
        print(f"\n--- {df_name} Numeric Summary ---\n")
        display(df.describe().style.set_table_styles([
            {'selector': 'th', 'props': [('background-color', 'lightblue'), ('color', 'black')]},
            {'selector': 'td', 'props': [('font-size', '10pt')]}
        ], overwrite=False))

    # --- Describe (categorical only) ---
    cat_cols = df.select_dtypes(exclude=np.number).columns
    if len(cat_cols) > 0:
        print(f"\n--- {df_name} Categorical Summary ---\n")
        cat_summary = df[cat_cols].describe().transpose()
        display(cat_summary.style.set_table_styles([
            {'selector': 'th', 'props': [('background-color', 'lightblue'), ('color', 'black')]},
            {'selector': 'td', 'props': [('font-size', '10pt')]}
        ], overwrite=False))

    print(f"\n{'='*80}\n")

# Apply
data_info(train, "train")
data_info(test, "test")



# Define excluded features
excluded_features = ['id', 'accident_risk'] # Exclude 'id' and the target variable

# Separate numerical and categorical features
numerical_features = [
    col for col in train.select_dtypes(include=np.number).columns
    if col not in excluded_features
]

categorical_features = [
    col for col in train.select_dtypes(exclude=np.number).columns
    if col not in excluded_features
]

# Print results
print(f"Numerical Features ({len(numerical_features)}): {numerical_features}")
print(f"Categorical Features ({len(categorical_features)}): {categorical_features}")


def plot_correlation_heatmap(df, numerical_cols, df_name, target='accident_risk', annot=True, threshold=0.5):
    """
    Generates and displays a correlation heatmap for specified numerical columns,
    highlighting features strongly correlated with each other and the target.
    """
    corr = df[numerical_cols].corr()

    # Sort columns by correlation with target (if present)
    if target in corr.columns:
        corr = corr.loc[corr.index, corr[target].abs().sort_values(ascending=False).index]

    # Create mask for upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    plt.figure(figsize=(12, 9))
    sns.heatmap(
        corr,
        mask=mask,
        annot=annot,
        cmap='RdYlGn',
        fmt=".2f",
        vmin=-1, vmax=1,
        cbar_kws={"shrink": .8}
    )
    plt.title(f'Correlation Heatmap of Numerical Features ({df_name})', fontsize=14)
    plt.show()

    # Print features strongly correlated with target
    if target in corr.columns:
        strong_corr = corr[target][(corr[target].abs() >= threshold) & (corr[target].abs() < 1)]
        if not strong_corr.empty:
            print(f"\nğŸ“Œ Features strongly correlated with {target} (|r| >= {threshold}):")
            print(strong_corr.sort_values(ascending=False))
        else:
            print(f"\nâ„¹ï¸� No strong correlations (|r| >= {threshold}) found with {target}.")

numerical_features_with_target = numerical_features + ['accident_risk']
plot_correlation_heatmap(train, numerical_features_with_target, "train")



def plot_numerical_distributions(train_df, test_df, numerical_cols):
    """
    Generates KDE and box plots for numerical features, comparing train vs test distributions,
    with summary statistics printed.
    """
    sns.set_style("whitegrid")
    sns.set_context("notebook")

    # Combine train and test for plotting
    combined_df = pd.concat([
        train_df[numerical_cols].assign(Source='Train'),
        test_df[numerical_cols].assign(Source='Test')
    ], axis=0, ignore_index=True)

    palette = ['#1f77b4', '#ff7f0e']  # Distinct colors for Train/Test

    for col in numerical_cols:
        # Summary Stats
        print(f"\nğŸ“Œ {col} Summary Statistics:")
        display(pd.DataFrame({
            'Train': [train_df[col].mean(), train_df[col].median(), train_df[col].std()],
            'Test': [test_df[col].mean(), test_df[col].median(), test_df[col].std()]
        }, index=['Mean', 'Median', 'Std']))

        fig, axes = plt.subplots(1, 2, figsize=(18, 6), gridspec_kw={'width_ratios': [2, 1]})

        # KDE Plot
        sns.kdeplot(
            data=combined_df, x=col, hue='Source', ax=axes[0], fill=True, palette="viridis"
        )
        axes[0].set_title(f'{col} Distribution (KDE)', fontsize=14)
        axes[0].set_xlabel('Density')
        axes[0].set_ylabel(col)

        # Box Plot
        sns.boxplot(
            data=combined_df, y=col, x='Source', ax=axes[1],
            orient='v', width=0.5, linewidth=1, fliersize=3, palette="viridis"
        )
        axes[1].set_title(f'{col} Boxplot', fontsize=14)
        axes[1].set_xlabel('Dataset')
        axes[1].set_ylabel(col)

        plt.tight_layout()
        plt.show()

# Call numerical distribution function
plot_numerical_distributions(train, test, numerical_features)



def plot_categorical_distributions(train_df, test_df, categorical_cols, target='accident_risk'):
    """
    Generates count plots for each categorical feature (train vs test)
    and bar plots showing mean target per category.
    Uses a denser layout with 2 plots per row.
    """
    if len(categorical_cols) == 0:
        print("No categorical features to plot.")
        return

    palette = ['#1f77b4', '#ff7f0e']  # Train / Test colors
    n_cols = 2
    n_rows = (len(categorical_cols) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, n_rows * 5))
    axes = axes.flatten()  # Flatten axes array for easy iteration

    for i, col in enumerate(categorical_cols):
        ax = axes[i]
        # Combine train and test for countplots
        combined = pd.concat([
            train_df[[col]].assign(Source='Train'),
            test_df[[col]].assign(Source='Test')
        ], axis=0, ignore_index=True)

        sns.countplot(x=col, hue='Source', data=combined, palette="viridis", ax=ax)
        ax.set_title(f'Distribution of {col} (Train vs Test)', fontsize=12)
        ax.set_xlabel(col)
        ax.set_ylabel('Count')
        ax.legend(title='Dataset')
        ax.tick_params(axis='x', rotation=45)

        # Overlay mean target per category as a line/barplot
        target_means = train_df.groupby(col)[target].mean().sort_values(ascending=False)
        ax2 = ax.twinx()
        sns.pointplot(x=target_means.index, y=target_means.values, ax=ax2, color='red', markers='o', linestyles='--')
        ax2.set_ylabel(f'Mean {target}', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

# Call the enhanced function
plot_categorical_distributions(train, test, categorical_features, target='accident_risk')

# Mean target per category overlay

# Plotted as a red dashed line with points.

# Helps you see which categories are associated with higher/lower accident risk



def plot_average_risk_by_category(train_df, categorical_cols, target_col):
    """
    Generates bar plots showing the average target value for each category
    in the specified categorical columns.
    """
    if len(categorical_cols) == 0:
        print("No categorical features to plot.")
        return

    n_cols = 2
    n_rows = (len(categorical_cols) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, n_rows * 5))
    axes = axes.flatten() # Flatten the axes array

    for i, col in enumerate(categorical_cols):
        ax = axes[i]
        # Calculate average target value per category
        avg_risk = train_df.groupby(col)[target_col].mean().sort_values()

        sns.barplot(x=avg_risk.index, y=avg_risk.values, ax=ax, palette='viridis')

        ax.set_title(f'Average {target_col} by {col}', fontsize=12)
        ax.set_xlabel(col)
        ax.set_ylabel(f'Average {target_col}')
        ax.tick_params(axis='x', rotation=45)

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

# Call the function
plot_average_risk_by_category(train, categorical_features, 'accident_risk')


plt.figure(figsize=(12, 5))

# KDE + Histogram overlay
sns.histplot(train['accident_risk'], bins=50, kde=True, color='skyblue', stat='density', alpha=0.6)

plt.title("Distribution of Accident Risk (Target Variable)", fontsize=14)
plt.xlabel("Accident Risk")
plt.ylabel("Density")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()



import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.model_selection import KFold
from sklearn.impute import SimpleImputer # Import SimpleImputer

class PreprocessingEnhanced:
    def __init__(self, train_df, test_df, target='accident_risk', missing=True, outliers=False, log_trf=False):
        self.train = train_df.copy()
        self.test = test_df.copy()
        self.target = target
        self.missing = missing
        self.outliers = outliers
        self.log_trf = log_trf
        self.imputers = {} # To store imputers for consistent transformation

    ##########################
    # 1ï¸�âƒ£ Memory Reduction
    ##########################
    @staticmethod
    def reduce_mem(df):
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64', "uint16", "uint32", "uint64"]
        for col in df.columns:
            col_type = df[col].dtypes
            if col_type in numerics:
                c_min = df[col].min()
                c_max = df[col].max()
                if "int" in str(col_type):
                    if c_min >= np.iinfo(np.int8).min and c_max <= np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min >= np.iinfo(np.int16).min and c_max <= np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min >= np.iinfo(np.int32).min and c_max <= np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    else:
                        df[col] = df[col].astype(np.int64)
                else:
                    if np.finfo(np.float16).min <= c_min <= c_max <= np.finfo(np.float16).max:
                        df[col] = df[col].astype(np.float16)
                    elif np.finfo(np.float32).min <= c_min <= c_max <= np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float64)
        return df

    ##########################
    # 2ï¸�âƒ£ Feature Creation
    ##########################
    @staticmethod
    def create_features(df):
        df = df.copy()

        # Polynomial features
        df['curvature_sq'] = df['curvature'] ** 2
        df['curvature_cubed'] = df['curvature'] ** 3
        df['speed_limit_sq'] = df['speed_limit'] ** 2

        # Interaction features
        df['speed_curvature'] = df['speed_limit'] * df['curvature']
        df['lanes_curvature'] = df['num_lanes'] * df['curvature']
        df['speed_lanes'] = df['speed_limit'] * df['num_lanes']
        df['accidents_speed'] = df['num_reported_accidents'] * df['speed_limit']
        df['accidents_curvature'] = df['num_reported_accidents'] * df['curvature']
        df['speed_accident'] = df['speed_limit'] * df['num_reported_accidents']
        # Handle potential division by zero in curvature_per_lane
        df['curvature_per_lane'] = df['curvature'] / (df['num_lanes'] + 1e-5)

        # Risk / binary features
        df['high_risk_combo'] = ((df['curvature'] > 0.5) & (df['speed_limit'] >= 60)).astype(int)
        df['weather_lighting_risk'] = ((df['weather'].isin(['foggy', 'rainy'])) &
                                       (df['lighting'].isin(['dim', 'night']))).astype(int)
        df['is_night'] = (df['lighting'] == 'night').astype(int)
        df['is_bad_weather'] = df['weather'].isin(['foggy', 'rainy']).astype(int)
        df['is_highway'] = (df['road_type'] == 'highway').astype(int)
        df['is_urban'] = (df['road_type'] == 'urban').astype(int)
        df['is_peak_time'] = df['time_of_day'].isin(['morning', 'evening']).astype(int)
        df['is_weekend'] = df['holiday'].astype(int)

        # Combined categorical features
        df['lighting_weather'] = df['lighting'].astype(str) + "_" + df['weather'].astype(str)
        df['weather_time'] = df['weather'].astype(str) + "_" + df['time_of_day'].astype(str)

        # Scores
        df['safety_score'] = df['road_signs_present'].astype(int) * 2 + \
                             (df['lighting'] == 'daylight').astype(int) + \
                             (df['weather'] == 'clear').astype(int)
        df['danger_score'] = (df['curvature'] > 0.6).astype(int) + \
                             (df['speed_limit'] >= 60).astype(int) + \
                             df['is_bad_weather'] + df['is_night'] + \
                             (df['num_reported_accidents'] >= 2).astype(int)

        # Ratio / log features
        df['accidents_per_lane'] = df['num_reported_accidents'] / (df['num_lanes'] + 1)
        df['risk_intensity'] = df['curvature'] * df['speed_limit'] / 50
        df['log_accidents'] = np.log1p(df['num_reported_accidents'])

        # Binning (handle potential NaNs from empty bins or boundaries)
        df['speed_bin'] = pd.cut(df['speed_limit'], bins=[0, 35, 50, 70, np.inf], labels=['low','medium','high','very_high'], right=False, include_lowest=True)
        df['curvature_bin'] = pd.qcut(df['curvature'], 4, labels=['very_low','low','high','very_high'], duplicates='drop')


        return df

    ##########################
    # 3ï¸�âƒ£ Prepare Data
    ##########################
    def prepare_data(self):
        self.X = self.train.drop([self.target, 'id'], axis=1, errors='ignore')
        self.y = self.train[self.target]
        self.test = self.test.drop('id', axis=1, errors='ignore')

    ##########################
    # 4ï¸�âƒ£ Target Encoding
    ##########################
    def target_encode(self, categorical_cols, n_splits=5, smoothing=1):
        print("Performing target encoding...")
        train_encoded = self.X.copy()
        test_encoded = self.test.copy()
        global_mean = self.y.mean()

        for col in categorical_cols:
            train_encoded[f"TE_{col}"] = np.nan
            test_encoded[f"TE_{col}"] = np.nan
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

            for train_idx, val_idx in kf.split(self.X):
                fold_train = self.X.iloc[train_idx]
                fold_y = self.y.iloc[train_idx]
                fold_val = self.X.iloc[val_idx]

                means = fold_y.groupby(fold_train[col]).mean()
                counts = fold_y.groupby(fold_train[col]).count()
                smooth = (means * counts + global_mean * smoothing) / (counts + smoothing)
                # Handle potential NaNs in mapping due to unseen categories
                train_encoded.iloc[val_idx, train_encoded.columns.get_loc(f"TE_{col}")] = fold_val[col].map(smooth).fillna(global_mean)

            # Handle potential NaNs in mapping due to unseen categories in test set
            full_means = self.y.groupby(self.X[col]).mean()
            test_encoded[f"TE_{col}"] = self.test[col].map(full_means).fillna(global_mean)

            # Drop original column
            train_encoded.drop(columns=[col], inplace=True)
            test_encoded.drop(columns=[col], inplace=True)

        self.X = train_encoded
        self.test = test_encoded

    ##########################
    # 5ï¸�âƒ£ Encode Categorical (Ordinal)
    ##########################
    def ordinal_encode_categorical(self):
        print("Performing ordinal encoding for remaining categorical features...")
        # Identify categorical columns that are still 'object' or 'category' dtype
        cat_cols_to_encode = self.X.select_dtypes(include=['object', 'category']).columns.tolist()

        if not cat_cols_to_encode:
            print("No object or category columns found for ordinal encoding.")
            return

        encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

        # Fit on combined train and test data to ensure all categories are seen
        combined_data = pd.concat([self.X[cat_cols_to_encode], self.test[cat_cols_to_encode]], axis=0)
        encoder.fit(combined_data)

        self.X[cat_cols_to_encode] = encoder.transform(self.X[cat_cols_to_encode])
        self.test[cat_cols_to_encode] = encoder.transform(self.test[cat_cols_to_encode])
        print(f"Ordinal encoded columns: {cat_cols_to_encode}")


    ##########################
    # 6ï¸�âƒ£ Encode Binary Flags
    ##########################
    def encode_binary_flags(self):
        print("Encoding binary/categorical flags...")
        flags = ['road_signs_present', 'public_road', 'holiday', 'school_season',
                 'is_night', 'is_bad_weather', 'is_highway', 'is_urban',
                 'is_peak_time', 'is_weekend', 'high_risk_combo',
                 'weather_lighting_risk', 'safety_score', 'danger_score']
        for flag in flags:
            if flag in self.X.columns:
                self.X[flag] = self.X[flag].astype(int)
                self.test[flag] = self.test[flag].astype(int)

    ##########################
    # 7ï¸�âƒ£ Handle Missing Values
    ##########################
    def handle_missing_values(self):
        print("Handling missing values...")
        # Identify columns with missing values after feature engineering/encoding
        cols_with_missing_train = self.X.columns[self.X.isnull().any()].tolist()
        cols_with_missing_test = self.test.columns[self.test.isnull().any()].tolist()
        all_cols_with_missing = list(set(cols_with_missing_train + cols_with_missing_test))

        if not all_cols_with_missing:
            print("No missing values found after preprocessing steps.")
            return

        print(f"Columns with missing values: {all_cols_with_missing}")

        # Impute numerical columns (all columns should be numerical after ordinal encoding)
        numeric_cols = self.X.select_dtypes(include=np.number).columns.tolist()
        cols_to_impute = [col for col in all_cols_with_missing if col in numeric_cols]

        if cols_to_impute:
            print(f"Imputing numerical columns ({len(cols_to_impute)}) with median...")
            imputer = SimpleImputer(strategy='median')
            # Fit on combined train and test data to ensure consistent imputation
            combined_data_to_impute = pd.concat([self.X[cols_to_impute], self.test[cols_to_impute]], axis=0)
            imputer.fit(combined_data_to_impute)

            self.X[cols_to_impute] = imputer.transform(self.X[cols_to_impute])
            self.test[cols_to_impute] = imputer.transform(self.test[cols_to_impute])
            print("Missing values imputed.")
        else:
            print("No numerical columns require imputation.")


    ##########################
    # 8ï¸�âƒ£ Scale Numeric Features
    ##########################
    def scale_numeric_features(self):
        print("Scaling numeric features...")
        numeric_cols = self.X.select_dtypes(include=np.number).columns.tolist()
        scaler = StandardScaler()
        self.X[numeric_cols] = scaler.fit_transform(self.X[numeric_cols])
        self.test[numeric_cols] = scaler.transform(self.test[numeric_cols])

    ##########################
    # 9ï¸�âƒ£ Fit Transform
    ##########################
    def fit_transform(self, target_encode_cols=None):
        # 1. Feature creation
        self.train = self.create_features(self.train)
        self.test = self.create_features(self.test)

        # 2. Prepare data
        self.prepare_data()

        # 3. Target encoding
        if target_encode_cols is not None:
            self.target_encode(target_encode_cols)

        # 4. Ordinal encode remaining categorical features
        self.ordinal_encode_categorical()

        # 5. Binary flag encoding
        self.encode_binary_flags()

        # 6. Handle Missing Values
        self.handle_missing_values()

        # 7. Scaling numeric features
        self.scale_numeric_features()

        # 8. Reduce memory
        self.X = self.reduce_mem(self.X)
        self.test = self.reduce_mem(self.test)

        # 9. Final check on dtypes - ensure no 'object' types remain
        print("Final dtypes after preprocessing:")
        print("Train dtypes:", self.X.dtypes.unique())
        print("Test dtypes:", self.test.dtypes.unique())


        # 10. Identify numeric/categorical columns after all encoding (should all be numeric)
        self.num_features = self.X.columns.tolist() # All columns should be numeric
        self.cat_features = [] # No categorical features remaining


        return self.X, self.y, self.test, self.cat_features, self.num_features

# How to call the PreprocessingEnhanced class:
# Instantiate with original train and test dataframes
prep = PreprocessingEnhanced(train, test)

# Define columns for target encoding (example)
target_encode_cols = ['road_type', 'weather', 'lighting', 'time_of_day']

# Fit and transform the data
X, y, test_processed, cat_features, num_features = prep.fit_transform(target_encode_cols)

print("\nPreprocessing complete.")
print(f"Processed X shape: {X.shape}")
print(f"Processed test shape: {test_processed.shape}")
print(f"Numeric features: {num_features}")
print(f"Categorical features: {cat_features}") # This will be empty after ordinal encoding
print(X.head())


import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from sklearn.base import BaseEstimator
import numpy as np

class KerasTabularModel(BaseEstimator):
    def __init__(self, embedding_dim_func=None,
                 hidden_units=[256,128,64],
                 dropout=0.3,
                 learning_rate=1e-3,
                 epochs=20,
                 batch_size=64,
                 cat_features=None,
                 num_features=None,
                 cat_cardinalities=None,
                 early_stopping_patience=3,
                 reduce_lr_patience=1,
                 task_type='regression',
                 n_classes=None):
        self.embedding_dim_func = embedding_dim_func or (lambda c: int(np.ceil(np.sqrt(c))))
        self.hidden_units = hidden_units
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.cat_features = cat_features or []
        self.num_features = num_features or []
        self.cat_cardinalities = cat_cardinalities or []
        self.early_stopping_patience = early_stopping_patience
        self.reduce_lr_patience = reduce_lr_patience
        self.task_type = task_type
        self.n_classes = n_classes

    def _build_model(self):
        # Categorical input
        cat_input = tf.keras.Input(shape=(len(self.cat_features),), name="cat_input")
        num_input = tf.keras.Input(shape=(len(self.num_features),), name="num_input")

        embs = []
        for j, card in enumerate(self.cat_cardinalities):
            emb_dim = int(np.ceil(np.sqrt(card)))
            xj = layers.Embedding(input_dim=card, output_dim=emb_dim)(cat_input[:,j])
            embs.append(layers.Flatten()(xj))

        x = layers.Concatenate()([num_input] + embs)
        for units in self.hidden_units:
            x = layers.Dense(units, activation='relu')(x)
            x = layers.Dropout(self.dropout)(x)
            x = layers.BatchNormalization()(x)

        if self.task_type == 'regression':
            output = layers.Dense(1, activation='linear')(x)
            loss = 'mse'
            metrics = ['mse']
        elif self.task_type == 'binary':
            output = layers.Dense(1, activation='sigmoid')(x)
            loss = 'binary_crossentropy'
            metrics = ['accuracy']
        else:
            output = layers.Dense(self.n_classes, activation='softmax')(x)
            loss = 'categorical_crossentropy'
            metrics = ['accuracy']

        self.model = Model(inputs=[cat_input, num_input], outputs=output)
        self.model.compile(optimizer=Adam(self.learning_rate), loss=loss, metrics=metrics)

    def _process_X(self, X):
        X_cat = X[self.cat_features].astype('int32').values if self.cat_features else np.zeros((len(X), 0))
        X_num = X[self.num_features].astype('float32').values if self.num_features else np.zeros((len(X), 0))
        return [X_cat, X_num]

    def fit(self, X, y, eval_set=None, verbose=1):
        self._build_model()
        X_proc = self._process_X(X)
        val_data = None
        if eval_set is not None:
            X_val, y_val = eval_set[0]
            X_val_proc = self._process_X(X_val)
            val_data = (X_val_proc, y_val)
        self.model.fit(X_proc, y,
                       validation_data=val_data,
                       epochs=self.epochs,
                       batch_size=self.batch_size,
                       verbose=verbose,
                       callbacks=[
                           tf.keras.callbacks.EarlyStopping(patience=self.early_stopping_patience, restore_best_weights=True),
                           tf.keras.callbacks.ReduceLROnPlateau(patience=self.reduce_lr_patience)
                       ])
        return self

    def predict(self, X):
        X_proc = self._process_X(X)
        preds = self.model.predict(X_proc)
        if self.task_type == 'regression':
            return preds.squeeze()
        elif self.task_type == 'binary':
            return (preds.squeeze() > 0.5).astype(int)
        else:
            return preds.argmax(axis=1)



import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.linear_model import Ridge, BayesianRidge, LinearRegression
import numpy as np

# Assuming KerasTabularModel is already defined and cat_features, num_features, cat_cardinalities are set
# from your previous code

# ------------------------------
# Global Config
# ------------------------------
random_state = 42

# ------------------------------
# LightGBM variants
# ------------------------------
lgb_models = {
    "LGBM_1": lgb.LGBMRegressor(
        boosting_type='gbdt',
        objective='regression',
        learning_rate=0.01,
        num_leaves=60,
        max_depth=8,
        n_estimators=5000,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
        n_jobs=-1,
        verbose=-1

        
    ),
    "LGBM_2": lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=7,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        verbose=-1
    )
}

# ------------------------------
# XGBoost variants
# ------------------------------
xgb_models = {
    "XGB_1": xgb.XGBRegressor(

        objective='reg:squarederror',
        learning_rate=0.01,
        max_depth=7,
        n_estimators=5000,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
        n_jobs=-1,
        verbosity=0
    ),
    "XGB_2": xgb.XGBRegressor(

        objective='reg:squarederror',
        learning_rate=0.01,
        enable_categorical =  True,
        max_depth=8,
        n_estimators=5000,
        booster =  'gbtree',
        early_stopping_rounds = 100,
        subsample = 0.8122972699153284, 
        reg_alpha = 3.4095237810895036e-08, 
        reg_lambda = 3.5877847439760364,
        colsample_bytree = 0.7345278109172685,
        random_state=random_state,
        n_jobs=-1,
        verbosity=0

        

    )
}

# ------------------------------
# CatBoost variants
# ------------------------------
cat_models = {
    "CAT_1": CatBoostRegressor(
        iterations=5000,
        learning_rate=0.01,
        depth=7,
        l2_leaf_reg=3,
        loss_function='RMSE',
        eval_metric='RMSE',
        verbose=False,
        random_seed=random_state
    ),
    "CAT_2": CatBoostRegressor(
        iterations=5000,
        learning_rate=0.01,
        depth=8,
        l2_leaf_reg=2,
        bagging_temperature=0.1,
        bootstrap_type='Bayesian',
        loss_function='RMSE',
        eval_metric='RMSE',
        verbose=False,
        random_seed=random_state
    )
}

# ------------------------------
# Linear / Bayesian stabilizers
# ------------------------------
linear_models = {
    "Ridge": Ridge(tol= 0.6006890587310936, 
                      alpha= 0.995272459379942, random_state=random_state),
    "BayesianRidge": BayesianRidge()
    
}

# Your categorical features
cat_features = ['lighting_weather', 'weather_time', 'speed_bin', 'curvature_bin']

# Compute cardinalities for embeddings
cat_cardinalities = [X[feat].nunique() for feat in cat_features]

# Numeric features
num_features = [col for col in X.columns if col not in cat_features]


# ------------------------------
# Neural Network (Keras Tabular)
# ------------------------------
from tensorflow.keras import layers, callbacks
nn_model = KerasTabularModel(
    hidden_units=[512, 256, 128],
    dropout=0.3,
    epochs=20,
    batch_size=100,
    cat_features=cat_features,
    num_features=num_features,
    cat_cardinalities=cat_cardinalities,
    early_stopping_patience=3,
    reduce_lr_patience=1
)

# ------------------------------
# Combine all base models
# ------------------------------
base_models = {}
base_models.update(lgb_models)
base_models.update(xgb_models)
base_models.update(cat_models)
base_models.update(linear_models)
base_models["NN"] = nn_model

# ------------------------------
# Summary
# ------------------------------
print(f"Total base models for stacking: {len(base_models)}")
for name in base_models:
    print(f"- {name}")



import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from tqdm import tqdm
import matplotlib.pyplot as plt
import lightgbm as lgb # Import lightgbm here


class LeaderboardTrainer:
    def __init__(self, X, y, X_test, models, task_type='regression', n_splits=5, random_state=42):
        """
        X, X_test        : Preprocessed data (can be used for both tree and linear models)
        y                : Target
        models           : Dictionary of model_name -> model instance
        task_type        : 'regression', 'binary', 'multiclass'
        n_splits         : Number of folds
        """
        self.X = X
        self.X_test = X_test
        self.y = y
        self.models = models
        self.task_type = task_type
        self.n_splits = n_splits
        self.random_state = random_state

        # Initialize folds
        if task_type == 'regression':
            self.folds = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        else:
            self.folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

        # Results
        self.scores = pd.DataFrame(columns=['Score'], dtype=float)
        self.OOF_preds = pd.DataFrame(dtype=float)
        self.TEST_preds = pd.DataFrame(dtype=float)

    # ---------------------------
    # Metric function
    # ---------------------------
    def score_metric(self, y_true, y_pred):
        if self.task_type == 'regression':
            return np.sqrt(mean_squared_error(y_true, y_pred))  # RMSE
        else:
            raise NotImplementedError("Classification metrics can be added")

    # ---------------------------
    # Fold-wise training
    # ---------------------------
    def train_single_model(self, model, model_name):
        oof_pred = np.zeros(self.X.shape[0], dtype=float)
        test_pred = np.zeros(self.X_test.shape[0], dtype=float)

        print("="*20)
        print(f"Training {model_name}")

        for fold, (train_idx, val_idx) in enumerate(self.folds.split(self.X, self.y)):
            print(f"Fold {fold+1}/{self.n_splits}")

            # Use the preprocessed X and X_test directly
            X_train, X_val = self.X.iloc[train_idx], self.X.iloc[val_idx]
            y_train, y_val = self.y.iloc[train_idx], self.y.iloc[val_idx]
            X_test = self.X_test # Use the full preprocessed test set

            # -------------------
            # Fit model
            # -------------------
            # Removed check for 'NN' model since KerasTabularModel is not defined
            if 'LGBM' in model_name:
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    eval_metric='rmse',
                    callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)] # Use callback for early stopping
                    # Removed early_stopping_rounds from here
                )
            elif 'XGB' in model_name:
                model.fit(X_train, y_train,
                        eval_set=[(X_val, y_val)], # Keep eval_set for monitoring (if verbosity allows)
                        verbose=False) # Keep verbose in fit if supported
            elif 'CAT' in model_name:
                model.fit(
                    X_train, y_train,
                    eval_set=(X_val, y_val),
                    early_stopping_rounds=50,
                    verbose=False
                )
            else: # For linear models (Ridge, LinearRegression, BayesianRidge)
                model.fit(X_train, y_train)

            # -------------------
            # Predictions
            # -------------------
            if self.task_type == 'regression':
                y_val_pred = model.predict(X_val)
                y_test_pred = model.predict(X_test)
            else:
                raise NotImplementedError("Classification support can be added")

            oof_pred[val_idx] = y_val_pred
            test_pred += y_test_pred / self.n_splits

            fold_score = self.score_metric(y_val, y_val_pred)
            print(f"Fold {fold+1} RMSE: {fold_score:.6f}")
            self.scores.loc[f"{model_name}", f"Fold_{fold+1}"] = fold_score

        self.scores.loc[f"{model_name}", "Score"] = self.scores.loc[f"{model_name}"][1:].mean()
        return oof_pred, test_pred

    # ---------------------------
    # Run all models
    # ---------------------------
    def run(self):
        for model_name, model in tqdm(self.models.items()):
            oof_pred, test_pred = self.train_single_model(model, model_name)
            self.OOF_preds[model_name] = oof_pred
            self.TEST_preds[model_name] = test_pred

        # ---------------------------
        # Meta-ensemble (stacking)
        # ---------------------------
        if len(self.models) > 1:
            print("\nTraining meta-model (Linear Regression stacking)...")
            meta_model = Ridge(alpha=1.0) # Using Ridge as meta-model
            oof_stack = self.OOF_preds.values
            test_stack = self.TEST_preds.values
            meta_model.fit(oof_stack, self.y)
            ensemble_pred = meta_model.predict(oof_stack)
            ensemble_test_pred = meta_model.predict(test_stack)
            self.OOF_preds["Ensemble"] = ensemble_pred
            self.TEST_preds["Ensemble"] = ensemble_test_pred

            ensemble_score = self.score_metric(self.y, ensemble_pred)
            self.scores.loc["Ensemble", "Score"] = ensemble_score
            print(f"Ensemble RMSE: {ensemble_score:.6f}")

        return self.TEST_preds

class MultiSeedTrainer(LeaderboardTrainer):
    def __init__(self, X, y, X_test, models, seeds=[42, 101, 2025], task_type='regression', n_splits=5):
        # Pass X and X_test directly to the parent class
        super().__init__(X, y, X_test, models, task_type, n_splits)
        self.seeds = seeds

    def run_multi_seed(self):
        # To store aggregated OOF and test predictions
        oof_agg = pd.DataFrame(dtype=float)
        test_agg = pd.DataFrame(dtype=float)

        for seed in self.seeds:
            print(f"\n=== Training with seed {seed} ===")
            self.random_state = seed
            self.folds = KFold(n_splits=self.n_splits, shuffle=True, random_state=seed)

            # Train all models for this seed
            seed_OOF_preds = pd.DataFrame(dtype=float)
            seed_TEST_preds = pd.DataFrame(dtype=float)

            for model_name, model in tqdm(self.models.items()):
                # Reset model instance to ensure seed is applied correctly for each seed run
                # Re-instantiate the model for each seed
                original_model_class = type(model)
                model_params = model.get_params()
                # Ensure random_state/random_seed is set for the new instance
                if 'random_state' in model_params:
                    model_params['random_state'] = seed
                if 'random_seed' in model_params:
                    model_params['random_seed'] = seed

                current_model = original_model_class(**model_params)


                oof_pred, test_pred = self.train_single_model(current_model, f"{model_name}_s{seed}")
                seed_OOF_preds[f"{model_name}_s{seed}"] = oof_pred
                seed_TEST_preds[f"{model_name}_s{seed}"] = test_pred

            oof_agg = pd.concat([oof_agg, seed_OOF_preds], axis=1)
            test_agg = pd.concat([test_agg, seed_TEST_preds], axis=1)

        # ---------------------------
        # Meta-ensemble across all seeds
        # ---------------------------
        print("\nTraining meta-model (stacking across seeds)...")
        meta_model = Ridge(alpha=1.0)
        meta_model.fit(oof_agg.values, self.y)
        ensemble_oof = meta_model.predict(oof_agg.values)
        ensemble_test = meta_model.predict(test_agg.values)

        self.OOF_preds = oof_agg
        self.TEST_preds = test_agg
        self.OOF_preds["Ensemble"] = ensemble_oof
        self.TEST_preds["Ensemble"] = ensemble_test

        ensemble_score = self.score_metric(self.y, ensemble_oof)
        print(f"\nFinal Ensemble RMSE (multi-seed): {ensemble_score:.6f}")

        return self.TEST_preds

# Instantiate the trainer with the preprocessed data
trainer = MultiSeedTrainer(
    X=X,
    y=y,
    X_test=test_processed, # Use test_processed as the test data
    models=base_models,
    seeds=[42],  # multi-seed  , 101, 2025
    n_splits=5
)

final_predictions = trainer.run_multi_seed()


import pandas as pd

def verify_submission(submission_file, sample_submission):
    """Perform comprehensive verification of submission format"""
    verification = pd.read_csv(submission_file)

    print("SUBMISSION VERIFICATION:")
    print(f"1. Submission shape: {verification.shape}")
    print(f"   Expected shape: {sample_submission.shape}")

    print(f"\n2. Submission columns: {verification.columns.tolist()}")
    print(f"   Expected columns: {sample_submission.columns.tolist()}")

    columns_match = verification.columns.tolist() == sample_submission.columns.tolist()
    print(f"\n3. Columns match exactly: {'âœ… YES' if columns_match else 'â�Œ NO'}")

    id_col = sample_submission.columns[0]
    id_match = set(verification[id_col]) == set(sample_submission[id_col])
    print(f"\n4. ID values match sample: {'âœ… YES' if id_match else 'â�Œ NO'}")

    target_col = sample_submission.columns[1]
    print(f"\n5. Target column statistics:")
    print(f"   Min: {verification[target_col].min():.2f}")
    print(f"   Max: {verification[target_col].max():.2f}")
    print(f"   Mean: {verification[target_col].mean():.2f}")
    print(f"   Std: {verification[target_col].std():.2f}")

    if columns_match and id_match:
        print("\nâœ… SUBMISSION FORMAT LOOKS CORRECT! Ready to upload.")
    else:
        print("\nâ�Œ SUBMISSION FORMAT HAS ISSUES! Please fix before uploading.")

    return verification


# Use the correct variables from the current notebook
# The variables 'test' and 'predictions' are already available from previous steps

# Use the final_ensemble_predictions from the stacked ensembling process
submission_df = pd.DataFrame({'id': test['id'], 'accident_risk': final_predictions["Ensemble"] })  #final_ensemble_predictions

submission_file = 'submission.csv'
# Assuming sample_submission is loaded correctly in cell IqsJbtsQr6he
# If not, you might need to adjust the path below
sample_submission_file = "/kaggle/input/playground-series-s5e10/sample_submission.csv"


try:
    sample_submission = pd.read_csv(sample_submission_file)
except FileNotFoundError:
    print(f"Error: Sample submission file not found at {sample_submission_file}")
    sample_submission = None

submission_df.to_csv(submission_file, index=False)

if sample_submission is not None:
  final_verification = verify_submission(submission_file, sample_submission)

  # This part is likely not needed if verify_submission passes,
  # but keeping it for robustness if needed.
  # It seems this block is unnecessary as verify_submission already prints the status.
  # if sample_submission is not None and final_verification.columns.tolist() != sample_submission.columns.tolist():
  #     print("\nAttempting to fix column names one last time based on sample submission...")
  #     final_verification.columns = sample_submission.columns
  #     final_verification.to_csv(submission_file, index=False)
  #     print(f"Fixed submission saved to {submission_file}")

  #     print("\nVerifying after attempting column name fix:")
  #     verify_submission(submission_file, sample_submission)


print("\nSubmission file created:")
display(submission_df.head())

