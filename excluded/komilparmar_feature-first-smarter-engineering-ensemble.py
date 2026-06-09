# ğŸ§ª Feature Engineering and Data Loading
import pandas as pd
import numpy as np
from sklearn.preprocessing import PowerTransformer

# Modeling
import cuml, cudf
import cupy as cp
from cuml.ensemble import RandomForestRegressor

# For Winsorization
from scipy.stats.mstats import winsorize

# For visualization
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.simplefilter('ignore')

train_df = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e5/sample_submission.csv")


def plot_percentiles(data, column, title=None, percentiles=None, test_data=None, 
                     highlight_percentiles=None):
    """
    Calculate percentiles, display statistics, and plot visualizations for a given column.
    
    Parameters:
    -----------
    data : pandas DataFrame
        The primary dataframe to analyze (training data)
    column : str
        Column name to analyze
    title : str, optional
        Custom title for the plots (defaults to column name)
    percentiles : list, optional
        List of percentiles to calculate (defaults to standard percentiles)
    test_data : pandas DataFrame, optional
        Optional test dataframe for comparison
    highlight_percentiles : list, optional
        Percentiles to highlight in the CDF plot (defaults to [25, 50, 75])
    """
    # Default percentiles if none provided
    if percentiles is None:
        percentiles = [0, 1, 5, 10, 25, 50, 75, 90, 95, 98, 99, 100]
    
    # Default highlight percentiles if none provided
    if highlight_percentiles is None:
        highlight_percentiles = [25, 50, 75]
    
    # Calculate percentiles for the provided column
    column_percentiles = np.percentile(data[column], percentiles)
    
    # Create DataFrame for display
    if test_data is not None:
        test_percentiles = np.percentile(test_data[column], percentiles)
        percentile_df = pd.DataFrame({
            'Percentile': percentiles,
            f'{column} (Train)': column_percentiles,
            f'{column} (Test)': test_percentiles
        })
    else:
        percentile_df = pd.DataFrame({
            'Percentile': percentiles,
            f'{column} Value': column_percentiles
        })
    
    # Display title and percentile table
    display_title = title if title else column
    print(f"{display_title} Percentiles:")
    display(percentile_df)
    
    # Prepare the figure
    if test_data is not None:
        # For comparison plots
        fig = plt.figure(figsize=(14, 10))
        
        # Create custom GridSpec
        gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.5])
        
        # Box plot - Train data
        ax1 = fig.add_subplot(gs[0, 0])
        sns.boxplot(ax=ax1, x=data[column])
        ax1.set_title(f'{display_title} Distribution (Train)')
        ax1.set_xlabel(column)
        
        # Box plot - Test data
        ax2 = fig.add_subplot(gs[0, 1])
        sns.boxplot(ax=ax2, x=test_data[column])
        ax2.set_title(f'{display_title} Distribution (Test)')
        ax2.set_xlabel(column)
        
        # CDF plot - Both datasets (spanning bottom row)
        ax3 = fig.add_subplot(gs[1, :])
        
        # Plot CDFs
        sns.ecdfplot(data=data, x=column, ax=ax3, label='Train', color='blue')
        sns.ecdfplot(data=test_data, x=column, ax=ax3, label='Test', color='red')
        
        ax3.set_title(f'Cumulative Distribution of {display_title}')
        ax3.set_xlabel(column)
        ax3.set_ylabel('Cumulative Probability')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # Add vertical lines for key percentiles on CDF plot
        # Training data percentiles
        train_highlight_values = np.percentile(data[column], highlight_percentiles)
        for p, v in zip(highlight_percentiles, train_highlight_values):
            ax3.axvline(x=v, linestyle='--', color='blue', alpha=0.6)
            ax3.text(v+0.5, 0.1, f'{p}th: {v:.1f}', rotation=90, color='blue', alpha=0.8)
        
        # Test data percentiles
        test_highlight_values = np.percentile(test_data[column], highlight_percentiles)
        for p, v in zip(highlight_percentiles, test_highlight_values):
            ax3.axvline(x=v, linestyle=':', color='red', alpha=0.6)
            ax3.text(v+0.5, 0.3, f'{p}th: {v:.1f}', rotation=90, color='red', alpha=0.8)
        
    else:
        # For single dataset
        fig, (ax_box, ax_cdf) = plt.subplots(2, 1, figsize=(10, 10))
        
        # Box plot
        sns.boxplot(ax=ax_box, x=data[column])
        ax_box.set_title(f'Box Plot of {display_title} Distribution')
        ax_box.set_xlabel(column)
        
        # CDF plot
        sns.ecdfplot(data=data, x=column, ax=ax_cdf)
        ax_cdf.set_title(f'Cumulative Distribution of {display_title}')
        ax_cdf.set_xlabel(column)
        ax_cdf.set_ylabel('Cumulative Probability')
        ax_cdf.grid(True, alpha=0.3)
        
        # Add vertical lines for key percentiles
        highlight_values = np.percentile(data[column], highlight_percentiles)
        for p, v in zip(highlight_percentiles, highlight_values):
            ax_cdf.axvline(x=v, linestyle='--', color='red', alpha=0.7)
            ax_cdf.text(v+0.5, 0.1, f'{p}th: {v:.1f}', rotation=90)
    
    plt.tight_layout()
    plt.show()
    
    return percentile_df  # Return DataFrame for further use if needed


def plot_kde_comparison(train_data, column, test_data=None, 
                       title=None, 
                       xlabel=None, 
                       plot_type='side_by_side',
                       train_color='cornflowerblue', 
                       test_color='indianred',
                       figsize=None):
    """
    Create KDE plots comparing a column between training and test datasets.
    
    Parameters:
    -----------
    train_data : pandas DataFrame
        The training dataframe
    test_data : pandas DataFrame
        The test dataframe
    column : str
        Column name to analyze
    title : str, optional
        Custom title base (default is column name)
    xlabel : str, optional
        Custom x-axis label (default is column name)
    plot_type : str, optional
        Type of plot: 'side_by_side' or 'overlay' (default: 'side_by_side')
    train_color : str, optional
        Color for training data (default: 'cornflowerblue')
    test_color : str, optional
        Color for test data (default: 'indianred')
    figsize : tuple, optional
        Figure size as (width, height) in inches
    """
    # Set default title and xlabel if not provided
    if title is None:
        title = f"{column} Distribution"
    if xlabel is None:
        xlabel = column
        
    # Set default figure sizes if not provided
    if figsize is None:
        figsize = (14, 6) if plot_type == 'side_by_side' else (10, 6)
    
    if test_data is None:
        plt.figure(figsize=figsize)
        
        sns.kdeplot(data=train_data, x=column, fill=True, color=train_color)
        plt.title(f'{title} in Training Data', fontsize=14)
        plt.xlabel(f'{xlabel}', fontsize=12)
        plt.ylabel('Density', fontsize=12)
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        return

    # Create side-by-side plots
    if plot_type == 'side_by_side':
        plt.figure(figsize=figsize)
        
        # Plot 1: Training data
        plt.subplot(1, 2, 1)
        sns.kdeplot(data=train_data, x=column, fill=True, color=train_color)
        plt.title(f'{title} in Training Data', fontsize=14)
        plt.xlabel(f'{xlabel}', fontsize=12)
        plt.ylabel('Density', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Test data
        plt.subplot(1, 2, 2)
        sns.kdeplot(data=test_data, x=column, fill=True, color=test_color)
        plt.title(f'{title} in Test Data', fontsize=14)
        plt.xlabel(f'{xlabel}', fontsize=12)
        plt.ylabel('Density', fontsize=12)
        plt.grid(True, alpha=0.3)
    
    # Create overlay plot
    elif plot_type == 'overlay':
        plt.figure(figsize=figsize)
        sns.kdeplot(data=train_data, x=column, label='Training Data', color=train_color)
        sns.kdeplot(data=test_data, x=column, label='Test Data', color=test_color)
        plt.title(f'{title}: Training vs Test Data', fontsize=15)
        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel('Density', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()
    
    plt.tight_layout()
    plt.show()


plot_kde_comparison(train_df, 'Age', test_data=test_df,)


percentiles_df = plot_percentiles(train_df, 'Age', test_data=test_df)


plot_kde_comparison(train_df, 'Height', test_data=test_df,)


percentiles_df = plot_percentiles(train_df, 'Height', test_data=test_df)


# winsorize Height by limiting 1% on both ends
# train_df['Height'] = winsorize(train_df['Height'], limits=[0.01, 0.01])
# test_df['Height'] = winsorize(test_df['Height'], limits=[0.01, 0.01])


plot_kde_comparison(train_df, 'Weight', test_df)


percentiles_df = plot_percentiles(train_df, 'Weight', test_data=test_df)


# train_df['Weight'] = winsorize(train_df['Weight'], limits=[0.005, 0.005])
# test_df['Weight'] = winsorize(test_df['Weight'], limits=[0.005, 0.005])


plot_kde_comparison(train_df, 'Duration', test_df)


percentiles_df = plot_percentiles(train_df, 'Duration', test_data=test_df)


plot_kde_comparison(train_df, 'Heart_Rate', test_df)


percentiles_df = plot_percentiles(train_df, 'Heart_Rate', test_data=test_df)


plot_kde_comparison(train_df, 'Body_Temp', test_df)


perpcentiles_df = plot_percentiles(train_df, 'Body_Temp', test_data=test_df)


plot_kde_comparison(train_df, column='Calories',)


percentiles_df = plot_percentiles(train_df, 'Calories')


# percentile_99 = np.percentile(train_df['Calories'], 99)

# # Drop outliers above 99th percentile
# train_df = train_df[train_df['Calories'] <= percentile_99]


train = train_df.copy()
test = test_df.copy()


# Make a column Sex_female such that there is a value 1 for female and -1 for male.
train['Sex_female'] = train['Sex'] == "female"
test['Sex_female'] = test['Sex'] == "female"

# Drop the original column
train = train.drop(columns=['Sex'])
test = test.drop(columns=['Sex'])

# Convert boolean to -1 and 1
train.loc[train['Sex_female']==True, "Sex_female"] = 1
train.loc[train['Sex_female']==False, "Sex_female"] = -1

test.loc[test['Sex_female']==True, "Sex_female"] = 1
test.loc[test['Sex_female']==False, "Sex_female"] = -1


# From training set
reflection_constant = train['Body_Temp'].max() + 1

# Apply the transformation to train
temp_reflected_train = reflection_constant - train['Body_Temp']
train['Body_Temp_transformed'] = np.log(temp_reflected_train)

# Apply the same transformation to test
temp_reflected_test = reflection_constant - test['Body_Temp']
test['Body_Temp_transformed'] = np.log(temp_reflected_test)


pt = PowerTransformer(method='yeo-johnson')
train['Body_Temp_transformed_YJ'] = pt.fit_transform(train['Body_Temp'].values.reshape(-1, 1))
test['Body_Temp_transformed_YJ'] = pt.transform(test['Body_Temp'].values.reshape(-1, 1))


# train['Body_Temp_dev'] = train['Body_Temp'] - 37.0
# test['Body_Temp_dev'] = test['Body_Temp'] - 37.0


# plot_kde_comparison(train, 'Body_Temp_dev', test_data=test, plot_type='overlay',)


# From training set
# reflection_constant = train['Body_Temp_dev'].max() + 1

# # Apply the transformation to train
# temp_reflected_train = reflection_constant - train['Body_Temp_dev']
# train['Body_Temp_dev_transformed'] = np.log(temp_reflected_train)

# # Apply the same transformation to test
# temp_reflected_test = reflection_constant - test['Body_Temp_dev']
# test['Body_Temp_dev_transformed'] = np.log(temp_reflected_test)


plot_kde_comparison(train, 'Body_Temp_transformed_YJ', test_data=test, plot_type='overlay')


# pt = PowerTransformer(method='yeo-johnson')
# train['Body_Temp_dev_transformed_YJ'] = pt.fit_transform(train['Body_Temp_dev'].values.reshape(-1, 1))
# test['Body_Temp_dev_transformed_YJ'] = pt.transform(test['Body_Temp_dev'].values.reshape(-1, 1))


# plot_kde_comparison(train, 'Body_Temp_dev_transformed_YJ', test_data=test, plot_type='overlay')


# Function to create BMI feature
def add_BMI(df):
    # Calculate BMI (assuming Height is in cm)
    df['Height_m'] = df['Height'] / 100  # Convert cm to meters
    df['BMI'] = df['Weight'] / (df['Height_m'] ** 2)
    
    # Drop the temporary Height_m column
    df.drop('Height_m', axis=1, inplace=True)
    
    return df

# Apply derived features to both datasets
train = add_BMI(train)
test = add_BMI(test)


# Calculate BMR using the Mifflin-St Jeor Equation
def add_BMR(df):
    # Calculate BMR for females
    df.loc[df['Sex_female']==1, 'BMR'] = 10 * df['Weight'] + 6.25 * df['Height'] - 5 * df['Age'] - 161
    # Calculate BMR for males
    df.loc[df['Sex_female']==-1, 'BMR'] = 10 * df['Weight'] + 6.25 * df['Height'] - 5 * df['Age'] + 5
    
    return df

train = add_BMR(train)
test = add_BMR(test)


numerical_features = train.columns[train.dtypes != 'object'].tolist()
categorical_features = train.columns[train.dtypes == 'object'].tolist()


numerical_features


numerical_features.remove('id')
numerical_features.remove('Calories')
numerical_features.append("Sex_female")


for col in numerical_features:
    train[col] = train[col].astype(np.float32)
    test[col] = test[col].astype(np.float32)


train_df = train.copy()
test_df = test.copy()


import pandas as pd
import numpy as np
import itertools
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures, StandardScaler

def add_feature_cross_terms(df, features):
    df = df.copy()
    df = df.loc[:, ~df.columns.duplicated()]  
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            f1 = features[i]
            f2 = features[j]
            df[f"{f1}_x_{f2}"] = df[f1] * df[f2]
    return df

def add_interaction_features(df, features):
    df_new = df.copy()
    for f1, f2 in itertools.combinations(features, 2):
        df_new[f"{f1}_plus_{f2}"] = df_new[f1] + df_new[f2]
        df_new[f"{f1}_minus_{f2}"] = df_new[f1] - df_new[f2]
        df_new[f"{f2}_minus_{f1}"] = df_new[f2] - df_new[f1]
        df_new[f"{f1}_div_{f2}"] = df_new[f1] / (df_new[f2] + 1e-5)
        df_new[f"{f2}_div_{f1}"] = df_new[f2] / (df_new[f1] + 1e-5)
    return df_new

def add_statistical_features(df, features):
    df_new = df.copy()
    df_new["row_mean"] = df[features].mean(axis=1)
    df_new["row_std"] = df[features].std(axis=1)
    df_new["row_max"] = df[features].max(axis=1)
    df_new["row_min"] = df[features].min(axis=1)
    df_new["row_median"] = df[features].median(axis=1)
    return df_new

train = add_feature_cross_terms(train, numerical_features)
test = add_feature_cross_terms(test, numerical_features)

train = add_interaction_features(train, numerical_features)
test = add_interaction_features(test, numerical_features)

train = add_statistical_features(train, numerical_features)
test = add_statistical_features(test, numerical_features)

poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
poly_train = poly.fit_transform(train[numerical_features])
poly_test = poly.transform(test[numerical_features])
poly_feature_names = poly.get_feature_names_out(numerical_features)

poly_train_df = pd.DataFrame(poly_train, columns=poly_feature_names)
poly_test_df = pd.DataFrame(poly_test, columns=poly_feature_names)

train = pd.concat([train.reset_index(drop=True), poly_train_df], axis=1)
test = pd.concat([test.reset_index(drop=True), poly_test_df], axis=1)

X = train.drop(columns=['id', 'Calories'])
y = np.log1p(train['Calories'])  
X_test = test.drop(columns=['id'])


FEATURES = X.columns.tolist()


import os
import joblib
from datetime import datetime

# Create directory for saved models if it doesn't exist
os.makedirs('saved_models', exist_ok=True)


from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_log_error
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import time

FOLDS = 8
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)
models = {
    'CatBoost': CatBoostRegressor(verbose=100, random_seed=42, early_stopping_rounds=100, task_type="GPU"),
    'XGBoost': XGBRegressor(tree_method='gpu_hist', predictor='gpu_predictor', max_depth=10, colsample_bytree=0.7, subsample=0.9, n_estimators=2000, learning_rate=0.02,
                            gamma=0.01, max_delta_step=2, early_stopping_rounds=100, eval_metric='rmse',
                            enable_categorical=True, random_state=42),
    'LightGBM': LGBMRegressor(n_estimators=2000, learning_rate=0.02, max_depth=10, colsample_bytree=0.7,
                              subsample=0.9, random_state=42, verbose=-1, device_type="gpu",),
    'RandomForest': RandomForestRegressor(
        n_estimators=500,
        max_depth=16,
        max_samples=0.4,
        max_features=16,
        min_samples_split=7,
        min_samples_leaf=3,
        # max_features='sqrt',
        random_state=42
    )
}

results = {name: {'oof': np.zeros(len(train)), 'pred': np.zeros(len(test)), 'rmsle': []} for name in models}

for name, model in models.items():
    print(f"\n=== Training {name} ===")
    for i, (train_idx, valid_idx) in enumerate(kf.split(X, y)):
        print(f"\nFold {i+1}")
        x_train, y_train = X.iloc[train_idx], y[train_idx]
        x_valid, y_valid = X.iloc[valid_idx], y[valid_idx]
        
        x_train = x_train.loc[:, ~x_train.columns.duplicated()]
        x_valid = x_valid.loc[:, ~x_valid.columns.duplicated()]
        x_test = X_test.loc[:, ~X_test.columns.duplicated()].copy()

        start = time.time()
        
        if name == 'XGBoost':
            model.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], verbose=100)
            model.save_model(f'saved_models/{name}_fold_{i+1}.json')
            # continue
        elif name == 'CatBoost':
            model.fit(x_train, y_train, eval_set=(x_valid, y_valid))
            model.save_model(f'saved_models/{name}_fold_{i+1}.cbm')
            # continue
        elif name == 'LightGBM':
            model.fit(x_train, y_train)
            model.booster_.save_model(f'saved_models/{name}_fold_{i+1}.txt')
            # continue
        else:
            model.fit(x_train, y_train)
            
            # Save with pickle for cuML models
            import pickle
            with open(f'saved_models/{name}_fold_{i+1}.pkl', 'wb') as f:
                pickle.dump(model, f)


        oof_pred = model.predict(x_valid)
        test_pred = model.predict(x_test)
        
        results[name]['oof'][valid_idx] = oof_pred
        results[name]['pred'] += test_pred / FOLDS
        
        rmsle = np.sqrt(mean_squared_log_error(np.expm1(y_valid), np.expm1(oof_pred)))
        results[name]['rmsle'].append(rmsle)
        
        print(f"Fold {i+1} RMSLE: {rmsle:.4f}")
        print(f"Training time: {time.time() - start:.1f} sec")


print("\n=== Model Comparison ===")
for name in models:
    mean_rmsle = np.mean(results[name]['rmsle'])
    std_rmsle = np.std(results[name]['rmsle'])
    print(f"{name} - Mean RMSLE: {mean_rmsle:.4f} Â± {std_rmsle:.4f}")


import numpy as np
import pandas as pd
import pickle
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from lightgbm import Booster
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import KFold

FOLDS = 8
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)

# Initialize result containers
results = {
    name: {'oof': np.zeros(len(train)), 'pred': np.zeros(len(test)), 'rmsle': []}
    for name in ['CatBoost', 'XGBoost', 'LightGBM', 'RandomForest']
}

# Dictionary to store all models per type
# all_models = {name: [] for name in results}

# Loop over folds and load each model
for i, (train_idx, valid_idx) in enumerate(kf.split(X, y)):
    print(f"\n=== Fold {i+1} ===")

    x_valid, y_valid = X.iloc[valid_idx], y[valid_idx]
    x_test = X_test.copy()

    x_valid = x_valid.loc[:, ~x_valid.columns.duplicated()]
    x_test = x_test.loc[:, ~x_test.columns.duplicated()]

    for name in results:
        model_path = f'saved_models/{name}_fold_{i+1}'
        
        if name == 'XGBoost':
            model = XGBRegressor()
            model.load_model(model_path + '.json')
            feature_names = model.get_booster().feature_names

        elif name == 'CatBoost':
            model = CatBoostRegressor()
            model.load_model(model_path + '.cbm')

        elif name == 'LightGBM':
            model = Booster(model_file=model_path + '.txt')

        elif name == 'RandomForest':
            with open(model_path + '.pkl', 'rb') as f:
                model = pickle.load(f)

        # Store the model
        # all_models[name].append(model) # Lots of memory usage

        # print(len(x_valid.columns))
        
        # Predict and store
        if name == 'LightGBM':
            oof_pred = model.predict(x_valid)
            test_pred = model.predict(x_test)
        elif name == 'XGBoost':
            oof_pred = model.predict(x_valid[feature_names])
            test_pred = model.predict(x_test[feature_names])
        else:
            oof_pred = model.predict(x_valid)
            test_pred = model.predict(x_test)

        results[name]['oof'][valid_idx] = oof_pred
        results[name]['pred'] += test_pred / FOLDS

        rmsle = np.sqrt(mean_squared_log_error(np.expm1(y_valid), np.expm1(oof_pred)))
        results[name]['rmsle'].append(rmsle)

        print(f"{name} RMSLE: {rmsle:.4f}")

# Show summary
print("\n=== Model Comparison (Loaded) ===")
for name in results:
    mean_rmsle = np.mean(results[name]['rmsle'])
    std_rmsle = np.std(results[name]['rmsle'])
    print(f"{name} - Mean RMSLE: {mean_rmsle:.4f} Â± {std_rmsle:.4f}")



from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_log_error

models = ['CatBoost', 'XGBoost', 'LightGBM', 'RandomForest']

# Step 1: Create OOF and test matrices for meta-model
oof_stack = np.column_stack([results[name]['oof'] for name in models])
test_stack = np.column_stack([results[name]['pred'] for name in models])
y_true = np.expm1(y)  # Back-transform ground truth

# Step 2: Train Ridge on OOF predictions (still in log space)
ridge = Ridge(alpha=5)
ridge.fit(oof_stack, y)

# Step 3: Predict on OOF and test data
oof_ensemble_log = ridge.predict(oof_stack)
test_ensemble_log = ridge.predict(test_stack)

# Step 4: Evaluate ensemble in original target space
rmsle_ensemble = np.sqrt(mean_squared_log_error(y_true, np.expm1(oof_ensemble_log)))
print(f"\n=== Ridge Ensemble RMSLE: {rmsle_ensemble:.4f} ===")

# Optional: Save ensemble prediction for submission
final_prediction = np.expm1(test_ensemble_log)


ridge.coef_


# Already done earlier:
# test_stack = np.column_stack([results[name]['pred'] for name in models])

# Predict log(Calories) using Ridge model
test_preds_log = ridge.predict(test_stack)

# Convert predictions back to original scale
test_preds = np.expm1(test_preds_log)  # This gives you the final prediction

# Optional: Show some predictions
print("\nSample Predictions:")
print(test_preds[:10])


submission = pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv')
submission['Calories'] = test_preds
submission.to_csv("/kaggle/working/submission.csv", index=False)


# model = XGBRegressor()
# model.load_model("saved_models/XGBoost_fold_1.json")


# Get feature importance
# importance_df = pd.DataFrame({
# 	'feature': model.get_booster().get_score(importance_type='weight').keys(),
# 	'importance': model.get_booster().get_score(importance_type='weight').values()
# })
# importance_df = importance_df.sort_values(by='importance', ascending=False)
# importance_df.reset_index(drop=True, inplace=True)
# importance_df['importance'] = importance_df['importance'] / importance_df['importance'].sum()  # Normalize


# # Find features containing 'row'
# row_features = importance_df[importance_df['feature'].str.contains('row', case=False)]

# # Add a rank column based on importance
# importance_df['rank'] = importance_df['importance'].rank(ascending=False, method='min')

# # Merge to get ranks of 'row' features
# row_feature_ranks = importance_df[importance_df['feature'].str.contains('row', case=False)][['feature', 'rank']]

# print(row_feature_ranks)


# import seaborn as sns
# sns.scatterplot(x=X['row_mean'], y=y)

