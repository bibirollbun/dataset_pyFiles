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


train = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/train.csv")
print("Train shape:",train.shape)
train.head()


import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Assuming train is already loaded
# train = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/train.csv")

# Prepare the data
def preprocess_data(df):
    X = df.drop(['efs','efs_time'], axis=1)
    y = df['efs']
    
    # Encode categorical variables
    le = LabelEncoder()
    categorical_cols = X.select_dtypes(include=['object']).columns
    
    for col in categorical_cols:
        X[col] = le.fit_transform(X[col].astype(str))
    
    # Handle missing values
    X = X.fillna(X.mean())
    
    return X, y

# Method 1: Random Forest Feature Importance
def get_rf_importance(X, y):
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    importance = pd.DataFrame({
        'feature': X.columns,
        'rf_importance': rf.feature_importances_
    })
    return importance

# Method 2: Mutual Information
def get_mi_importance(X, y):
    mi_scores = mutual_info_classif(X, y, random_state=42)
    importance = pd.DataFrame({
        'feature': X.columns,
        'mi_importance': mi_scores
    })
    return importance

# Method 3: Correlation with target
def get_correlation_importance(X, y):
    if y.dtype == 'object':
        le = LabelEncoder()
        y_numeric = le.fit_transform(y)
    else:
        y_numeric = y
    
    correlations = []
    for column in X.columns:
        corr = np.corrcoef(X[column], y_numeric)[0,1]
        correlations.append(abs(corr))
    
    importance = pd.DataFrame({
        'feature': X.columns,
        'corr_importance': correlations
    })
    return importance

# Method 4: XGBoost Feature Importance
def get_xgb_importance(X, y):
    # If y is categorical, encode it
    if y.dtype == 'object':
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
    else:
        y_encoded = y
    
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    xgb_model.fit(X, y_encoded)
    
    importance = pd.DataFrame({
        'feature': X.columns,
        'xgb_importance': xgb_model.feature_importances_
    })
    return importance

# Combine all methods and get top 50%
def get_top_features(train):
    # Preprocess data
    X, y = preprocess_data(train)
    
    # Get importance from all methods
    rf_imp = get_rf_importance(X, y)
    mi_imp = get_mi_importance(X, y)
    corr_imp = get_correlation_importance(X, y)
    xgb_imp = get_xgb_importance(X, y)
    
    # Merge all importance scores
    combined = rf_imp.merge(mi_imp, on='feature').merge(corr_imp, on='feature').merge(xgb_imp, on='feature')
    
    # Normalize scores to 0-1 scale
    for col in ['rf_importance', 'mi_importance', 'corr_importance', 'xgb_importance']:
        combined[col] = (combined[col] - combined[col].min()) / (combined[col].max() - combined[col].min())
    
    # Calculate average importance
    combined['avg_importance'] = combined[['rf_importance', 'mi_importance', 'corr_importance', 'xgb_importance']].mean(axis=1)
    
    # Sort by average importance
    combined = combined.sort_values('avg_importance', ascending=False)
    
    # Get top 50%
    n_features = len(combined)
    n_top = int(n_features * 0.5)
    top_features = combined.head(n_top)
    
    return top_features




# Execute and display results
print("Train shape:", train.shape)
top_features = get_top_features(train)

print("\nTop 50% Most Important Features:")
top_features[['feature', 'avg_importance', 'rf_importance', 'mi_importance', 'corr_importance', 'xgb_importance']]

# Optional: Save results
# top_features.to_csv('top_features.csv', index=False)




