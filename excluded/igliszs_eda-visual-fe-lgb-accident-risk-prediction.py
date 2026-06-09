import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


from typing import Dict, List

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import optuna

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error

from lightgbm import LGBMRegressor


sample_sub = pd.read_csv('/kaggle/input/playground-series-s5e10/sample_submission.csv')
train_df = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')


train_df = train_df.drop(columns=['id'])
test_df = test_df.drop(columns=['id'])


train_df.duplicated().sum()


train_df = train_df.drop_duplicates()
train_df.duplicated().sum()


train_df.shape, test_df.shape, sample_sub.shape


print("Data gaps Ğ² train:", train_df.isnull().sum())
print("\nData gaps Ğ² test:", test_df.isnull().sum())


train_df.describe()


g = sns.catplot(
    data=train_df, kind="bar",
    x="lighting", y="accident_risk", hue='weather',
    palette="dark", alpha=.5, height=5
)
g.set_axis_labels("Lighting Conditions", "Accident Risk Level")
plt.title("Impact of Lighting and Weather Conditions on Accident Risk")
plt.xticks(rotation=45)
plt.show()

# Description:
# """
# The chart displays the relationship between lighting conditions, weather factors, and road accident risk levels.
# Bars are grouped by lighting types with color coding representing different weather conditions.
# The analysis helps identify the most dangerous combinations of environmental factors.
# """


g = sns.catplot(
    data=train_df, kind="bar",
    x="road_type", y="accident_risk", hue='lighting',
    palette="dark", alpha=.6, height=6
)
g.set_axis_labels("Road Type", "Accident Risk Level")
plt.title("Impact of Road Type and Lighting Conditions on Accident Risk")
plt.xticks(rotation=45)
# plt.tight_layout()
plt.show()

# Description:
# """
# The chart displays the relationship between road type, lighting conditions, and accident risk levels.
# Bars are grouped by road types with color coding representing different lighting conditions.
# Analysis reveals which road types are most affected by lighting conditions and identifies high-risk combinations.
# """


g = sns.catplot(
    data=train_df, kind="bar",
    x="time_of_day", y="accident_risk", hue='road_type',
    palette="dark", alpha=.6, height=4
)
g.set_axis_labels("Time of Day", "Accident Risk Level")
plt.title("Accident Risk by Time of Day and Road Types")
plt.xticks(rotation=45)
plt.show()

# Description:
# """
# Analysis of road accident risk variations throughout the day across different road types. 
# The chart reveals how risk levels fluctuate during daily periods and identifies 
# high-risk time segments for specific road infrastructure categories, highlighting 
# temporal patterns in road safety.
# """


def plt_analysis(df, columns, figsize=(14, 6), palette='viridis'):
    """
    Enhanced function for analyzing categorical variables distribution
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame for analysis
    columns : list
        List of column names to analyze
    figsize : tuple, optional
        Figure size (default: (14, 6))
    palette : str, optional
        Color palette for plots (default: 'viridis')
        
    Returns:
    --------
    None
        Displays distribution analysis plots for each specified column
    """
    for column in columns:
        if column not in df.columns:
            print(f"âš ï¸� Column '{column}' not found in DataFrame")
            continue
            
        plt.figure(figsize=figsize)
        
        # Left plot: Countplot
        plt.subplot(1, 2, 1)
        ax1 = sns.countplot(data=df, x=column, palette=palette,
                            hue=column,
                           order=df[column].value_counts().index)
        plt.title(f'DISTRIBUTION: {column.upper()}', fontweight='bold', pad=20)
        plt.xlabel(column.title())
        plt.ylabel('NUMBER OF OBSERVATIONS')
        plt.xticks(rotation=45)
        plt.grid(axis='y', alpha=0.3)
        
        # Add count annotations on bars
        for p in ax1.patches:
            ax1.annotate(f'{p.get_height():.0f}', 
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='bottom', fontweight='bold')
        
        # Right plot: Pie chart
        plt.subplot(1, 2, 2)
        value_counts = df[column].value_counts()
        colors = sns.color_palette(palette, len(value_counts))
        
        wedges, texts, autotexts = plt.pie(value_counts.values, 
                                          labels=value_counts.index,
                                          autopct='%1.1f%%',
                                          colors=colors,
                                          startangle=90)
        plt.title(f'PROPORTIONS: {column.upper()}', fontweight='bold', pad=20)
        
        # Improve pie chart text readability
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(10)
        
        plt.tight_layout()
        plt.show()
        
        # Display statistics
        print(f"ğŸ“Š STATISTICS FOR '{column}':")
        print(f"   Unique values: {df[column].nunique()}")
        print(f"   Most frequent: {df[column].mode().iloc[0]} ({value_counts.iloc[0]} observations)")
        print(f"   Total observations: {len(df)}")
        print("-" * 50)


numeric_features = train_df.select_dtypes(include=['number']).columns
object_features = train_df.select_dtypes(include=['object', 'bool']).columns
numeric_features, object_features


plt_analysis(train_df, object_features)


corr_matrix = train_df.select_dtypes(include=[np.number]).corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix')
plt.show()


def scalar_encoder(df, numeric_col, object_col, scaler_type='standard', encoder_type='LE', return_scalers=False):
    """
    Unified function for scaling numerical features and encoding categorical variables.
    
    Performs data preprocessing by applying scaling to numerical columns and encoding 
    to categorical columns in a single function call.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame containing the data to be processed
    numeric_col : list
        List of numerical column names to be scaled
    object_col : list  
        List of categorical column names to be encoded
    scaler_type : str, optional
        Type of scaler to use for numerical features:
        - 'standard': StandardScaler (zero mean, unit variance)
        - 'minmax': MinMaxScaler (scale to [0, 1] range)
        - 'robust': RobustScaler (robust to outliers)
        - 'No': No scaling applied
        Default: 'standard'
    encoder_type : str, optional
        Type of encoder to use for categorical features:
        - 'LE': LabelEncoder (integer encoding)
        - 'OE': OrdinalEncoder (integer encoding for multiple columns)
        - 'OHE': OneHotEncoder (binary columns for each category)
        - 'No': No encoding applied
        Default: 'LE'
    return_scalers : bool, optional
        If True, returns both transformed data and fitted transformers dictionary
        If False, returns only transformed data
        Default: False
        
    Returns:
    --------
    tuple or pandas.DataFrame
        If return_scalers=True: (transformed_data, transformers_dict)
        If return_scalers=False: transformed_data
        
        transformers_dict contains fitted scaler/encoder objects for each column
        
    Raises:
    -------
    ValueError
        If scaler_type or encoder_type is not in supported options
        
    Examples:
    ---------
    >>> # Basic usage with default parameters
    >>> processed_data = scalar_encoder(df, ['age', 'income'], ['category', 'city'])
    
    >>> # With MinMax scaling and OneHot encoding
    >>> processed_data = scalar_encoder(df, ['age'], ['category'], 
    ...                                scaler_type='minmax', encoder_type='OHE')
    
    >>> # Get transformers for later use on test data
    >>> train_processed, transformers = scalar_encoder(train_df, num_cols, cat_cols, 
    ...                                               return_scalers=True)
    """
    
    # Available scalers mapping
    scalers = {
        'standard': StandardScaler,
        'minmax': MinMaxScaler,
        'robust': RobustScaler,
        'No': None
    }

    # Available encoders mapping
    encoders = {
        'LE': LabelEncoder,      # Integer encoding for single column
        'OE': OrdinalEncoder,    # Integer encoding for multiple columns  
        'OHE': OneHotEncoder,    # Binary columns for each category
        'No': None
    }
    
    # Validate input parameters
    if scaler_type not in scalers:
        raise ValueError(f"scaler_type must be one of: {list(scalers.keys())}")
    if encoder_type not in encoders:
        raise ValueError(f"encoder_type must be one of: {list(encoders.keys())}")
    
    # Create copy to avoid modifying original data
    transform_data = df.copy()
    transforms_dict = {}  # Store fitted transformers

    # Get selected scaler and encoder classes
    scaler_class = scalers[scaler_type]
    encoder_class = encoders[encoder_type]
    
    # Apply scaling to numerical columns
    if scaler_class is not None:
        for column in numeric_col:
            if column not in transform_data.columns:
                continue
            
            # Initialize and fit scaler
            scaler = scaler_class()
            # Scale data and flatten from 2D to 1D array
            transform_data[column] = scaler.fit_transform(transform_data[[column]]).flatten()
            # Store fitted scaler for potential inverse transformation
            transforms_dict[column] = scaler

    # Apply encoding to categorical columns
    if encoder_class is not None:
        for column in object_col:
            if column not in transform_data.columns:
                continue
            
            encoder = encoder_class()
            
            # Handle different encoding types
            if encoder_type == 'OHE':
                # OneHotEncoder creates multiple binary columns
                encoded = encoder.fit_transform(transform_data[[column]])
                # Convert sparse matrix to DataFrame with meaningful column names
                encoded_df = pd.DataFrame(encoded.toarray(), 
                                        columns=[f"{column}_{cat}" for cat in encoder.categories_[0]],
                                        index=transform_data.index)
                # Replace original column with encoded columns
                transform_data = pd.concat([transform_data.drop(columns=[column]), encoded_df], axis=1)
                
            elif encoder_type == 'LE':
                # LabelEncoder for single column
                transform_data[column] = encoder.fit_transform(transform_data[column])
                
            else:
                # OrdinalEncoder for consistent encoding
                transform_data[column] = encoder.fit_transform(transform_data[[column]]).flatten()
            
            # Store fitted encoder
            transforms_dict[column] = encoder
    
    # Return based on return_scalers flag
    return (transform_data, transforms_dict) if return_scalers else transform_data


# drop target and male X,y and test for model
X = train_df.copy().drop(columns='accident_risk')
y = train_df['accident_risk']
test = test_df.copy()


def create_optimized_features(df):
    """
    Create optimized features
    
    Parameters:
    df : pandas.DataFrame
    -----------
    Returns:
    df : pandas.DataFrame
    """
    df = df.copy()

    df['log_speed'] = np.log1p(df['speed_limit'])
    df['log_accidents'] = np.log1p(df['num_reported_accidents'])
    df['log_curvature'] = np.log1p(df['curvature'])
    
    return df


# create new features
X_tr = create_optimized_features(X)
test_tr = create_optimized_features(test)


# repeat, because we dropped 'accident_risk' and made new features
numeric_features = X_tr.select_dtypes(include=['number']).columns
object_features = X_tr.select_dtypes(include=['object', 'bool']).columns
numeric_features, object_features


# transform train and test data
X_tr = scalar_encoder(X_tr, numeric_features, object_features, scaler_type='robust', encoder_type='LE')
test_tr = scalar_encoder(test_tr, numeric_features, object_features, scaler_type='robust', encoder_type='LE')
# P.S. 'robust' and 'LE' transform make max pred score


# params from Optuna
LGB_best_params = {
    'n_estimators': 2700,
    'learning_rate': 0.014291084943047132,
    'num_leaves': 200,
    'max_depth': 12,
    'min_child_samples': 11,
    'min_child_weight': 0.002,
    'subsample': 0.7378723574719579,
    'subsample_freq': 1,
    'colsample_bytree': 0.9223783646573634,
    'reg_alpha': 8.003237221691328e-08,
    'reg_lambda': 1.9726751536952425e-05,
    'min_split_gain':  0.004,
    'feature_fraction': 0.9 ,
}


# align indices for KFold - prevents indexing errors during cross-validation
X_reset = X_tr.reset_index(drop=True)
y_reset = y.reset_index(drop=True)


# initialize LightGBM model with optimized hyperparameters
# configured for regression task with RMSE metric and parallel processing
LGB_model = LGBMRegressor(**LGB_best_params,
                          objective='regression',
                          metric='rmse',
                          boosting_type='gbdt',
                          random_state=42,
                          n_jobs=-1,
                          verbose=-1)


# 5-fold cross-validation setup with shuffling for robust evaluation 
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []

# iterates through all data splits and calculates RMSE for each fold
for fold, (train_idx, val_idx) in enumerate(kfold.split(X_reset), 1):
    X_tr, X_val = X_reset.iloc[train_idx], X_reset.iloc[val_idx]
    y_tr, y_val = y_reset.iloc[train_idx], y_reset.iloc[val_idx]
    
    # train LightGBM
    LGB_model.fit(X_tr, y_tr)
    
    # predict on validation fold
    y_pred = LGB_model.predict(X_val)

    # calculate RMSE for this fold
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    cv_scores.append(rmse)
    print(f"Fold {fold}: {rmse:.5f}")

# calculate final cross-validation performance
# mean RMSE across all folds with standard deviation
simple_avg_score = np.mean(cv_scores)
print(f"\nSimple Average CV Score: {simple_avg_score:.5f} (+/- {np.std(cv_scores):.5f})")


# train on full data and predict
LGB_model.fit(X_reset, y_reset)

sub_pred = LGB_model.predict(test_tr)

sample_sub['accident_risk'] = sub_pred
sample_sub


sample_sub.to_csv("submission.csv", index=False)

