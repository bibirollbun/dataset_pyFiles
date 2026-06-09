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


pip install autoviz


pip install autogluon


import pandas as pd
import numpy as np
import autoviz
from autoviz import AutoViz_Class
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from autogluon.core.metrics import make_scorer
from scipy import stats
from autogluon.features.generators import AutoMLPipelineFeatureGenerator
from autogluon.tabular import TabularDataset, TabularPredictor


# Load the datasets
train_data = pd.read_csv(r"/kaggle/input/rohlik-sales-forecasting-challenge-v2/sales_train.csv")
test_data =  pd.read_csv(r"/kaggle/input/rohlik-sales-forecasting-challenge-v2/sales_test.csv")
inventory = pd.read_csv(r"/kaggle/input/rohlik-sales-forecasting-challenge-v2/inventory.csv")
calendar = pd.read_csv(r"/kaggle/input/rohlik-sales-forecasting-challenge-v2/calendar.csv")
solution_file = pd.read_csv(r"/kaggle/input/rohlik-sales-forecasting-challenge-v2/solution.csv")
test_weights = pd.read_csv(r"/kaggle/input/rohlik-sales-forecasting-challenge-v2/test_weights.csv")


def explore_datasets(train_data, test_data, inventory, calendar, solution_file, test_weights):
    """
    Explore and visualize characteristics of datasets using AutoViz
    
    Parameters:
    - Multiple pandas DataFrames from the Rohlik Sales Forecasting Challenge
    """
    # Initialize AutoViz
    av = AutoViz_Class()
    
    # List of datasets to explore
    datasets = {
        'Train Data': train_data,
        'Test Data': test_data,
        'Inventory': inventory,
        'Calendar': calendar,
        'Solution File': solution_file,
        'Test Weights': test_weights
    }
    
    # Exploration results dictionary
    exploration_results = {}
    
    # Explore each dataset
    for name, df in datasets.items():
        print(f"\n--- {name} Dataset Characteristics ---")
        
        # Basic information
        print(f"Shape: {df.shape}")
        print("\nColumn Types:")
        print(df.dtypes)
        
        print("\nMissing Values:")
        missing_values = df.isnull().sum()
        print(missing_values[missing_values > 0] if any(missing_values > 0) else "No missing values")
        
        print("\nSummary Statistics:")
        print(df.describe().T)
        
        # Generate AutoViz visualizations (optional, as it might create multiple files)
        try:
            output_filename = f"{name.lower().replace(' ', '_')}_autoviz.html"
            av.AutoViz(
                filename="", 
                dfte=df,
                header=0, 
                verbose=0,
                max_rows_analyzed=100000,
                max_cols_analyzed=30,
                save_plot_dir='autoviz_output',
                chart_format='bokeh'
            )
            print(f"\nAutoViz HTML report generated: autoviz_output/{output_filename}")
        except Exception as e:
            print(f"Could not generate AutoViz visualization for {name}: {e}")
    
    return datasets
    
# Run exploration
datasets = explore_datasets(
    train_data.sample(frac=0.2,random_state=12), test_data, inventory, 
    calendar, solution_file, test_weights
)




# Joining datasets
def prepare_data(train_data, test_data, inventory, calendar, test_weights):
    train_merged = pd.merge(train_data, inventory, on=['unique_id', 'warehouse'], how='left')
    train_merged = pd.merge(train_merged, calendar, on=['date', 'warehouse'], how='left')
    
    test_merged = pd.merge(test_data, inventory, on=['unique_id', 'warehouse'], how='left')
    test_merged = pd.merge(test_merged, calendar, on=['date', 'warehouse'], how='left')
    
    return train_merged, test_merged

# WMAE Calculation Function
    
train_merged, test_merged = prepare_data(
    train_data, test_data, inventory, calendar, test_weights
)

# Use test weights
test_weights_dict = dict(zip(test_weights['unique_id'], test_weights['weight']))




def prepare_features(df):
    # Make a copy to avoid modifying original dataframe
    df = df.copy()
    
    # 1. Categorical Encoding
    le = LabelEncoder()
    categorical_cols = [
        'warehouse', 'L1_category_name_en', 'L2_category_name_en', 
        'L3_category_name_en', 'L4_category_name_en', 'holiday_name'
    ]
    
    for col in categorical_cols:
        if col in df.columns:
            df[col + '_encoded'] = le.fit_transform(df[col].astype(str).fillna('Unknown'))
    df=df.drop(columns=categorical_cols)
    print("-----------")
    print(df.columns)
    print("-----------")

    
    # 2. Date Features
    df['date'] = pd.to_datetime(df['date'])
    date_features = [
        'day_of_week', 'month', 'year', 
        'day_of_month', 'week_of_year', 
        'quarter', 'is_month_start', 'is_month_end'
    ]
    
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['day_of_month'] = df['date'].dt.day
    df['week_of_year'] = df['date'].dt.isocalendar().week
    df['quarter'] = df['date'].dt.quarter
    df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
    df['is_month_end'] = df['date'].dt.is_month_end.astype(int)
    
    # 3. Holiday and Special Day Features
    holiday_cols = ['holiday', 'shops_closed', 'winter_school_holidays', 'school_holidays']
    for col in holiday_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(int)
    
    # 4. Discount Features
    discount_cols = [col for col in df.columns if 'discount' in col]
    if discount_cols:
        # Total discount
        df['total_discount'] = df[discount_cols].sum(axis=1)
        # Max discount
        df['max_discount'] = df[discount_cols].max(axis=1)
        # Discount variance
        df['discount_variance'] = df[discount_cols].var(axis=1)
    
    # 6. Interaction Features
    if 'sell_price_main' in df.columns and 'total_discount' in df.columns:
        df['price_after_discount'] = df['sell_price_main'] * (1 - df['total_discount']/100)
    
    # 7. Cyclical Encoding for Time Features
    df['sin_day_of_week'] = np.sin(df['day_of_week'] * (2 * np.pi / 7))
    df['cos_day_of_week'] = np.cos(df['day_of_week'] * (2 * np.pi / 7))
    df['sin_month'] = np.sin(df['month'] * (2 * np.pi / 12))
    df['cos_month'] = np.cos(df['month'] * (2 * np.pi / 12))
    
    # # 8. Optional: PCA for Dimensionality Reduction of Discount Features
    # if len(discount_cols) > 1:
    #     pca = PCA(n_components=1)
    #     df['discount_pca'] = pca.fit_transform(df[discount_cols])
    
    return df

def lgbm_imputer(train_data, test_data, target_column='sales'):
    # Prepare features
    train_prep = prepare_features(train_data.copy())
    test_prep = prepare_features(test_data.copy())
    print("+++++++++++")
    print(train_prep.columns)
    # Identify columns for imputation
    numeric_cols = train_prep.select_dtypes(include=['float64', 'int64']).columns
    
    # Columns to use as features for imputation
    feature_cols = [col for col in numeric_cols]
    
    # Identify missing columns in test data
    missing_cols = test_prep.columns[test_prep.isnull().any()].tolist()
    
    # Impute for each column with missing values
    for col in missing_cols:
        # Separate data with and without missing values
        train_known = train_prep[train_prep[col].notnull()]
        train_missing = train_prep[train_prep[col].isnull()]
        
        # Prepare features and target
        X_known = train_known[feature_cols]
        y_known = train_known[col]
        
        # Prepare test data features for imputation
        X_missing = train_missing[feature_cols]

        # Train LGBM model
        model = lgb.LGBMRegressor(random_state=42)
        model.fit(X_known, y_known)
        
        # Predict missing values
        imputed_values = model.predict(X_missing)
        
        # Fill missing values
        train_prep.loc[train_prep[col].isnull(), col] = imputed_values
        
        # Impute test data if needed
        test_missing = test_prep[test_prep[col].isnull()]
        if not test_missing.empty:
            X_test_missing = test_missing[feature_cols]
            test_imputed_values = model.predict(X_test_missing)
            test_prep.loc[test_prep[col].isnull(), col] = test_imputed_values
        
    return train_prep, test_prep

# Usage
train_imputed, test_imputed = lgbm_imputer(train_merged.drop(['unique_id'],axis=1).copy(), test_merged.drop(['unique_id'],axis=1).copy())


train_imputed['total_orders']=train_imputed['total_orders'].ffill()
train_imputed['sales']=train_imputed['sales'].ffill()


def explore_datasets(train_data, test_data):
    """
    Explore and visualize characteristics of datasets using AutoViz
    
    Parameters:
    - Multiple pandas DataFrames from the Rohlik Sales Forecasting Challenge
    """
    # Initialize AutoViz
    av = AutoViz_Class()
    
    # List of datasets to explore
    datasets = {
        'Train Data': train_data,
        'Test Data': test_data
    }
    
    # Exploration results dictionary
    exploration_results = {}
    
    # Explore each dataset
    for name, df in datasets.items():
        print(f"\n--- {name} Dataset Characteristics ---")
        
        # Basic information
        print(f"Shape: {df.shape}")
        print("\nColumn Types:")
        print(df.dtypes)
        
        print("\nMissing Values:")
        missing_values = df.isnull().sum()
        print(missing_values[missing_values > 0] if any(missing_values > 0) else "No missing values")
        
        print("\nSummary Statistics:")
        print(df.describe().T)
        
        # Generate AutoViz visualizations (optional, as it might create multiple files)
        try:
            output_filename = f"{name.lower().replace(' ', '_')}_autoviz.html"
            av.AutoViz(
                filename="", 
                dfte=df,
                header=0, 
                verbose=0,
                depVar="sales",
                max_rows_analyzed=100000,
                max_cols_analyzed=40,
                save_plot_dir='autoviz_output',
                chart_format='bokeh'
            )
            print(f"\nAutoViz HTML report generated: autoviz_output/{output_filename}")
        except Exception as e:
            print(f"Could not generate AutoViz visualization for {name}: {e}")
    
    return datasets
    
# Run exploration
datasets = explore_datasets(
    train_imputed, test_imputed)


# Basic CatBoost configurati
hyperparameters = {
    'GBM': [  # This includes CatBoost, LightGBM, and XGBoost
        {
            'model_type': 'CatBoost'  # Specifically use CatBoost
        }
    ]
}

# More detailed CatBoost configuration
advanced_hyperparameters = {
    'GBM': [
        {
            'model_type': 'CatBoost',
            'learning_rate': 0.1,
            'iterations': 100,
            'depth': 6,
            'l2_leaf_reg': 3,
            'random_strength': 1,
            'min_data_in_leaf': 20,
            'num_boost_round':100
        }
    ]
}

# Example usage in model training
predictor = TabularPredictor(label='sales', 
                              eval_metric='mean_absolute_error',
                             problem_type='regression'
                            )

# Fit the model
predictor.fit(
    train_data=train_imputed.drop(['availability'],axis=1),
    hyperparameters=hyperparameters,  # or advanced_hyperparameters
    time_limit=600,  # optional: limit training time to 10 minutes
    verbosity=2,
    presets = ['medium']
)

# Make predictions
predictions = predictor.predict(test_imputed)


df_preds=pd.DataFrame(predictions)
df_preds.columns=['sales_hat']


df=pd.concat([solution_file['id'],df_preds],axis=1)


df.to_csv('submission.csv',index=False)




