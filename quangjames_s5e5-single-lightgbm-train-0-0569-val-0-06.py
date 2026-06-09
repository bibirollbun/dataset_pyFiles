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


import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder,PowerTransformer
from sklearn.model_selection import KFold, train_test_split
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor


train_dataset = '/kaggle/input/playground-series-s5e5/train.csv'
test_dataset = '/kaggle/input/playground-series-s5e5/test.csv'

train_data = pd.read_csv(train_dataset)
test_data = pd.read_csv(test_dataset)

print(train_data.info())
print(train_data.head(5))

print(test_data.info())
print(test_data.head(5))


# Check outiine values
numeric_cols = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp','Calories']
# Check for infinity (inf)
for col in numeric_cols:
    has_inf = train_data[col].isin([float('inf'), -float('inf')]).sum()
    has_nan = train_data[col].isna().sum()
    print(f"Column {col}: {has_inf} Inf values:, {has_nan} NaN values")

# Outliers boxplot 
plt.figure(figsize=(12, 6))
train_data[numeric_cols].boxplot()
plt.title('Outlier Boxplots of Numeric''s columns')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# Data preparation
train_df_processed = train_data.copy().drop('id', axis=1)
test_ids = test_data['id']  
test_df_processed = test_data.copy().drop('id', axis=1)

# Outlier handling for Calories (only train)
q1 = train_df_processed['Calories'].quantile(0.25)
q3 = train_df_processed['Calories'].quantile(0.75)
iqr = q3 - q1
upper_bound = q3 + 1.5 * iqr
lower_bound = q1 - 1.5 * iqr
train_df_processed['Calories'] = train_df_processed['Calories'].clip(lower=lower_bound, upper=upper_bound)

# Outlier handling for Body_Temp (both train and test)
q1 = train_df_processed['Body_Temp'].quantile(0.25)
q3 = train_df_processed['Body_Temp'].quantile(0.75)
iqr = q3 - q1
upper_bound = q3 + 1.5 * iqr
lower_bound = q1 - 1.5 * iqr
train_df_processed['Body_Temp'] = train_df_processed['Body_Temp'].clip(lower=lower_bound, upper=upper_bound)
test_df_processed['Body_Temp'] = test_df_processed['Body_Temp'].clip(lower=lower_bound, upper=upper_bound)

# Create new features
def add_features(df):
    df['Heart_Rate_Duration'] = df['Heart_Rate'] * df['Duration']
    df['Age_Weight'] = df['Age'] * df['Weight']
    # BMI (cm)
    df['BMI'] = df['Weight'] / ((df['Height'] / 100) ** 2)
    df['Duration_Age'] = df['Duration'] * df['Age']    
    df['Duration_Weight'] = df['Duration'] * df['Weight']      
    df['Duration_Height'] = df['Duration'] * df['Height']
    df['Duration_Height'] = df['Duration'] * df['Height']   
    df['Heart_Rate_BMI'] = df['Heart_Rate'] * df['BMI']
    df['Duration_Squared'] = df['Duration'] ** 2
    df['Heart_Rate_Squared'] = df['Heart_Rate'] ** 2
    df['Heart_Rate_Duration_Squared'] = df['Heart_Rate_Duration'] ** 2
    return df

train_df_processed = add_features(train_df_processed)
test_df_processed = add_features(test_df_processed)

# numeric_features and categorical_features definition
numeric_features = [col for col in train_df_processed.select_dtypes(include=['int64', 'float64']).columns.tolist() if col != 'Calories'] 
numeric_features = [feat for feat in numeric_features if feat not in ['Height', 'BMI', 'Weight', 'Heart_Rate_BMI', 'Body_Temp']]
#print(numeric_features)
#numeric_features = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp', 'Heart_Rate_Duration', 
#                    'Age_Weight', 'BMI', 'Duration_Age', 'Heart_Rate_BMI', 'Duration_Squared', 'Heart_Rate_Squared']
categorical_features = ['Sex']

# X and y definition
X = train_df_processed.drop('Calories', axis=1)
y = np.log1p(train_df_processed['Calories'])  #Log transformation for Calories
X_test = test_df_processed  

# Preprocessor definition
preprocessor = ColumnTransformer(
    transformers=[
        ('num', PowerTransformer(method='yeo-johnson'), numeric_features),
        ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
    ])

# Pipeline definition
pipeline = Pipeline(steps=[('preprocessor', preprocessor)])

# Fit và transform
X = pipeline.fit_transform(X)
X_test = pipeline.transform(X_test)

# Check size
#print(f"Column of train: {X.columns()}")
print(f"Size of train: {X.shape}")
print(f"Size of test: {X_test.shape}")

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Size of train for training: {X_train.shape}")
print(f"Size of validation for training: {X_val.shape}")


# RMSLE function (modified to match log transformed y)
def rmsle(y_true_log, y_pred_log):
    # Return y_true and y_pred to their original values
    y_true = np.expm1(y_true_log)  # expm1 to invert log1p
    y_pred = np.expm1(y_pred_log)
    # Make sure there are no negative values
    y_true = np.clip(y_true, 0, None)
    y_pred = np.clip(y_pred, 0, None)
    log_true = np.log1p(y_true)
    log_pred = np.log1p(y_pred)
    return np.sqrt(np.mean(np.square(log_pred - log_true)))

# Custom Evaluation Function for LightGBM
def rmsle_eval(preds, dataset):
    y_true_log = dataset.get_label()
    score = rmsle(y_true_log, preds)
    return 'rmsle', score, False

# Custom callback to save RMSLE history
class RMSLEHistoryCallback:
    def __init__(self):
        self.train_rmsle_history = []
        self.val_rmsle_history = []
    
    def __call__(self, env):
        iteration = env.iteration
        if iteration == 0:
            self.train_rmsle_history = []
            self.val_rmsle_history = []
        for data_name, eval_name, eval_result, _ in env.evaluation_result_list:
            if data_name == 'train' and eval_name == 'rmsle':
                self.train_rmsle_history.append(eval_result)
            elif data_name == 'val' and eval_name == 'rmsle':
                self.val_rmsle_history.append(eval_result)

# KFold Setup
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# RMSLE Archive List
train_rmsle_scores = []
val_rmsle_scores = []

# List of models from each fold (for ensemble)
models = []

# LightGBM Parameter Settings
params = {
    'objective': 'regression',
    'num_leaves': 50,
    'min_child_samples' : 30,
    'learning_rate': 0.05,
    'feature_fraction': 0.7,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'lambda_l1': 2,
    'lambda_l2': 1,
    'random_state': 42,
    'device': 'gpu',  
    'gpu_platform_id': 0,  
    'gpu_device_id': 0,    
    'verbose': -1
}

# Training with KFold
fold = 1
for train_idx, val_idx in kf.split(X):
    print(f"\nFold {fold}")
    
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    rmsle_callback = RMSLEHistoryCallback()
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=15000,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'val'],
        feval=rmsle_eval,
        callbacks=[lgb.log_evaluation(period=200), lgb.early_stopping(stopping_rounds=500), rmsle_callback]
    )
    models.append(model)
    
    # Predict and calculate RMSLE
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    
    train_rmsle = rmsle(y_train, y_train_pred)
    val_rmsle = rmsle(y_val, y_val_pred)
    
    train_rmsle_scores.append(train_rmsle)
    val_rmsle_scores.append(val_rmsle)
    
    print(f"Fold {fold} - Train RMSLE: {train_rmsle:.4f}, Val RMSLE: {val_rmsle:.4f}")
    
    # Plot the graph after each fold
    plt.figure(figsize=(8, 6))
    plt.plot(rmsle_callback.train_rmsle_history, label='Train RMSLE', linestyle='--')
    plt.plot(rmsle_callback.val_rmsle_history, label='Val RMSLE')
    plt.title(f'Training Progress (RMSLE) - Fold {fold}')
    plt.xlabel('Iteration')
    plt.ylabel('RMSLE')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    fold += 1

# Calculate the average RMSLE
mean_train_rmsle = np.mean(train_rmsle_scores)
mean_val_rmsle = np.mean(val_rmsle_scores)
print(f"\nMean Train RMSLE: {mean_train_rmsle:.4f}")
print(f"Mean Val RMSLE: {mean_val_rmsle:.4f}")
print("Training os completed!")
# Huấn luyện mô hình cuối trên toàn bộ dữ liệu train
#final_train_data = lgb.Dataset(X, label=y)
#final_model = lgb.train(
#    params,
#    final_train_data,
#    num_boost_round=model.best_iteration if hasattr(model, 'best_iteration') else 4000
#)

# Ensemble: Average prediction on the test set from models of folds



lgb.plot_importance(model, max_num_features=10)
plt.show()

# Get the importance of all features
feature_importance = model.feature_importance(importance_type='gain')

feature_names = numeric_features + list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features))

importance_list = list(zip(feature_names, feature_importance))

importance_list.sort(key=lambda x: x[1], reverse=True)

print("Feature Importance List (All Features):")
for feature, importance in importance_list:
    print(f"{feature}: {importance}")

print("\nTop 10 Most Important Features:")
for feature, importance in importance_list[:10]:
    print(f"{feature}: {importance}")


y_test_preds = []
for model in models:
    y_test_pred = np.expm1(model.predict(X_test))
    y_test_preds.append(y_test_pred)
y_test_pred = np.mean(y_test_preds, axis=0)
y_test_pred = np.clip(y_test_pred, 0, None)
submission = pd.DataFrame({'id': test_ids, 'Calories': y_test_pred})
submission.to_csv('submission.csv', index=False)

print('Predict and save the submission were completed!')

