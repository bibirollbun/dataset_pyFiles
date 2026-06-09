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


#pip install pandas numpy scikit-learn lightgbm


import pandas as pd
import numpy as np
import lightgbm as lgb
import holidays
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')


# 1. Load Data
train = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')


# In this specific competition, 'num_sold' has nulls that should be removed
train = train.dropna(subset=['num_sold']).reset_index(drop=True)


# 2. Advanced Feature Engineering
def engineer_features(df):
    df['date'] = pd.to_datetime(df['date'])
    
    # Basic components
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['dayofweek'] = df['date'].dt.dayofweek
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
    
    # Cyclical encoding
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['day_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
    
    # Holiday logic
    df['is_holiday'] = 0
    unique_countries = df['country'].unique()
    country_map = {'Canada': 'CA', 'Estonia': 'EE', 'Japan': 'JP', 'Spain': 'ES', 'Argentina': 'AR'}
    
    for country in unique_countries:
        h_code = country_map.get(country, 'US')
        country_holidays = holidays.CountryHoliday(h_code)
        mask = df['country'] == country
        df.loc[mask, 'is_holiday'] = df.loc[mask, 'date'].apply(lambda x: 1 if x in country_holidays else 0)

    # Special peaks
    df['last_week_dec'] = ((df['month'] == 12) & (df['day'] >= 24)).astype(int)
    return df

print("Engineering features...")
train = engineer_features(train)
test = engineer_features(test)


# 3. Categorical Encoding (FIXED: 'product' instead of 'item')
cat_cols = ['country', 'store', 'product'] 
for col in cat_cols:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col])
    test[col] = le.transform(test[col])


# 4. Target Transformation (Crucial for MAPE)
y = np.log1p(train['num_sold'])
X = train.drop(['id', 'date', 'num_sold'], axis=1)
X_test = test.drop(['id', 'date'], axis=1)


# 5. Training with LightGBM
tscv = TimeSeriesSplit(n_splits=5)
test_preds = np.zeros(len(X_test))

lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'verbosity': -1,
    'boosting_type': 'gbdt',
    'random_state': 42,
    'learning_rate': 0.05,
    'num_leaves': 63,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.7,
    'max_depth': -1,
}

print("Starting Cross-Validation...")
for train_index, val_index in tscv.split(X):
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]
    
    model = lgb.LGBMRegressor(**lgb_params, n_estimators=3000)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=100), lgb.log_evaluation(period=500)]
    )
    
    # Accumulate predictions in original scale
    test_preds += np.expm1(model.predict(X_test)) / tscv.get_n_splits()


# 6. Final Post-processing
submission = pd.DataFrame({
    'id': test['id'],
    'num_sold': np.round(test_preds)
})

submission.to_csv('submission.csv', index=False)
print("Success! Submission saved.")


import matplotlib.pyplot as plt
import seaborn as sns

# Set visual style
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (15, 8)

# 1. Visualize Sales Trends over Time
def plot_sales_trends(df):
    plt.figure(figsize=(15, 6))
    # Group by date and country for clarity
    trend = df.groupby(['date', 'country'])['num_sold'].sum().reset_index()
    sns.lineplot(data=trend, x='date', y='num_sold', hue='country')
    plt.title('Daily Sticker Sales by Country', fontsize=16)
    plt.ylabel('Total Units Sold')
    plt.show()

# 2. Visualize Target Distribution (Log vs Original)
def plot_distributions(df):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    sns.histplot(df['num_sold'], bins=50, kde=True, ax=axes[0], color='blue')
    axes[0].set_title('Original num_sold Distribution')
    
    sns.histplot(np.log1p(df['num_sold']), bins=50, kde=True, ax=axes[1], color='green')
    axes[1].set_title('Log Transformed num_sold (What the model sees)')
    plt.show()

# 3. Seasonal Patterns
def plot_seasonality(df):
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    
    # Day of Week
    sns.boxplot(data=df, x='dayofweek', y='num_sold', ax=axes[0])
    axes[0].set_title('Sales by Day of Week (0=Mon, 6=Sun)')
    
    # Monthly
    sns.barplot(data=df, x='month', y='num_sold', ax=axes[1])
    axes[1].set_title('Average Sales by Month')
    plt.show()

# 4. Feature Importance (From your trained LightGBM model)
def plot_feature_importance(model, X):
    feature_imp = pd.DataFrame({
        'Value': model.feature_importances_,
        'Feature': X.columns
    }).sort_values(by="Value", ascending=False)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x="Value", y="Feature", data=feature_imp.head(15))
    plt.title('Top 15 Most Important Features for Prediction')
    plt.show()

# 5. Correlation Heatmap
def plot_correlation(df):
    plt.figure(figsize=(12, 10))
    # Select numeric columns only
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    sns.heatmap(corr, annot=False, cmap='coolwarm', linewidths=0.5)
    plt.title('Feature Correlation Matrix')
    plt.show()

# --- EXECUTE VISUALIZATIONS ---
# Note: Use original train data (before encoding) for better labels if possible, 
# or use the transformed 'train' dataframe for feature importance.

print("Generating Visualizations...")
plot_distributions(train)
plot_sales_trends(train)
plot_seasonality(train)
plot_correlation(train)
plot_feature_importance(model, X) # 'model' is your last trained LGBM model from the previous code

