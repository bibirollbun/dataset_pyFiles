# Import packages         
import numpy as np           
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt 
from scipy.stats import pointbiserialr

import lightgbm as lgb
from lightgbm import LGBMClassifier, early_stopping 
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score 
from sklearn.model_selection import TimeSeriesSplit
from catboost import CatBoostClassifier
from sklearn.ensemble import VotingClassifier
import shap

pd.set_option('display.max_columns', None)

# Import packages for warnings
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))             


# Load Data
train_df = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')     


train_df.head() 


test_df.head()


print('Train:', train_df.shape)
print('Test:', test_df.shape)


# Check duplicate rows 
print('Duplicate rows in Train Dataset:', train_df.duplicated().sum())
print('Duplicate rows in Test Dataset:', train_df.duplicated().sum())


print('Train:')
train_df.info()
print()
print('Test')
test_df.info()   


# Correct the column name `temparature` to `temperature`
train_df.rename(columns={'temparature': 'temperature'}, inplace=True)
test_df.rename(columns={'temparature': 'temperature'}, inplace=True)      


# Check column names
print('Train Dataset:\n', train_df.columns)
print('Test Dataset:\n', test_df.columns)


# Check for data inconsistencies in `day` column 
print('Check for Data inconsistencies in `day` column in train_df')
print(train_df['day'].value_counts().value_counts())   
print()
print('Check for Data inconsistencies in `day` column in test_df')
print(test_df['day'].value_counts().value_counts())   


# Histogram for `day` column
sns.histplot(train_df['day'], bins=365);       


# Create a new feature `corrected_day` to ensure each year has sequential days from 1 to 365 
train_df['corrected_day'] = (np.arange(len(train_df)) % 365) + 1

# Identify and display rows where the original `day` column differs from `corrected_day` 
train_df[train_df['day'] != train_df['corrected_day']][['day', 'corrected_day']].head() 


# Examine a specific range of rows to observe data inconsistencies 
train_df.loc[1035:1039, ['day', 'corrected_day']]


# Check if any days from 1 to 365 are missing in the original `day` column 
missing_days = set(range(1, 366)) - set(train_df['day'].unique())
missing_days   # Returns an empty set if no days are missing                                       


# Overwrite the `day` column with `corrected_day` to fix inconsistencies 
train_df['day'] = train_df['corrected_day']

# Verify that each day (1-365) appears exactly 6 times in the dataset 
train_df['day'].value_counts().value_counts()


# Drop `corrected_day` column
train_df.drop(columns=['corrected_day'], inplace=True)   


# Check Missing Values 
print('Missing Values in Train Dataset:', train_df.isna().sum().sum())
print()
print('Test Dataset Values in Test Dataset:')
test_df.isna().sum()           


# Impute missing value with median            
test_df.fillna(test_df['winddirection'].median(), inplace=True)

# Verify if there is any missing values
print('Missing values in Test Dataset:', test_df.isna().sum().sum())


# Create donut chart to display rainfall distribution
# Aggregate data for the rainfall column
rainfall_count = train_df['rainfall'].value_counts()

# Prepare labels and values
labels = ['Rainfall', 'No Rainfall']
sizes = rainfall_count
color = ['#FFB3BA', '#BAFFC9']

plt.figure(figsize=(6, 6))
plt.pie(sizes, labels=labels, startangle=90, autopct='%1.1f%%',
        colors=color,
        wedgeprops={'width':0.4})
plt.title('Rainfall Distribution', weight='bold')
plt.show()     


col = train_df.drop(columns=['id', 'rainfall', 'day']).columns

# Setup subplots
fig, axes = plt.subplots(len(col), 2, figsize=(13, 5 * len(col)))

# Plot histogram for train_df and test_df
for i, var in enumerate(col):
    axes[i, 0].hist(train_df[var], alpha=0.5, label='Train')
    axes[i, 0].hist(test_df[var], alpha=0.5, label='Test')
    axes[i, 0].set_title(f'Histogram for {var}', weight='bold')
    axes[i, 0].legend()

    # Prepare data for boxplot
    combined = pd.concat([train_df[var].to_frame().assign(dataset='Train'),
                          test_df[var].to_frame().assign(dataset='Test')])    

    # Plot boxplot
    sns.boxplot(data=combined, x='dataset', y=var, ax=axes[i, 1], palette='Set2')
    axes[i, 1].set_title(f'Boxplot for {var}', weight='bold')

plt.tight_layout()
plt.show()   


def kde_hist(col_name):          
    # Setup subplots
    fig, axes = plt.subplots(1, 2, figsize=(13, 5 * 1))
    # Create KDE plot
    sns.kdeplot(data=train_df, x=col_name, hue='rainfall', common_norm=False, ax=axes[0])
    axes[0].set_title(f'KDE Plot for {col_name}', weight='bold')
    # Create Histogram 
    sns.histplot(data=train_df, x=col_name, hue='rainfall', multiple='fill', ax=axes[1])  
    axes[1].set_title(f'Histogram for {col_name} by Rainfall', weight='bold')
    axes[1].set_ylabel('%', rotation=0, weight='bold') 
    plt.show()
    # Descriptive stats for each feature by `rainfall`
    stats = train_df.groupby('rainfall')[col_name].describe()   
    return stats


kde_hist('cloud')  


kde_hist('sunshine')     


kde_hist('humidity')     


kde_hist('pressure')    


kde_hist('temperature')     


kde_hist('mintemp')  


kde_hist('maxtemp') 


kde_hist('dewpoint')


kde_hist('windspeed')  


kde_hist('winddirection') 


# Create month
def map_day_to_month(day):    
    if day <= 31:
        return 1
    elif day <= 59:
        return 2
    elif day <= 90:
        return 3
    elif day <= 120:
        return 4
    elif day <= 151:
        return 5
    elif day <= 181:
        return 6
    elif day <= 212:
        return 7
    elif day <= 243:
        return 8
    elif day <= 273:
        return 9
    elif day <= 304:
        return 10
    elif day <= 334:
        return 11
    else:
        return 12

# Mapping days to corresponding months
train_df['month'] = train_df['day'].apply(map_day_to_month)
test_df['month'] = test_df['day'].apply(map_day_to_month)


# Calculate rainfall percentage by month
monthly_rainfall = train_df.groupby('month')['rainfall'].mean().mul(100).round()   
monthly_rainfall = monthly_rainfall.sort_values(ascending=False)

# Setup subplots
fig, axes = plt.subplots(1, 2, figsize=(12, 4 * 1))

# Create barplot
a = sns.barplot(x=monthly_rainfall.index.astype(str), 
                y=monthly_rainfall.values, palette = 'Blues_r', ax=axes[0])
axes[0].set_title('Percentage of Rainy Days per Month', weight='bold')
axes[0].set_ylabel('Rainfall Percentage (%)', weight='bold')

# Add labels on the top of the bar
for bars in a.containers:
    a.bar_label(bars)   

# Create scatterplot
sns.scatterplot(data=train_df, x='day', y='rainfall', hue='rainfall', ax=axes[1])
axes[1].set_title('Rainfall Distribution Over the Days of the Year', weight='bold');     


# Create heatmap        
plt.figure(figsize=(15, 8))
sns.heatmap(train_df.drop(columns=['id', 'day']).corr(), annot=True, cmap='viridis')
plt.title('Correlation Heatmap for Train Dataset', weight='bold');                


binary_target = 'rainfall'  
numerical_columns = ['sunshine', 'cloud', 'humidity', 'windspeed']

# Calculate point biserial correlation
results = {}
for col in numerical_columns:
    correlation, p_value = pointbiserialr(train_df[binary_target], train_df[col])
    results[col] = {'Correlation': correlation, 'P_value': p_value}

# Convert results to a DataFrame
results_df = pd.DataFrame(results)
results_df        


#exclude_col = ['id', 'day', 'rainfall', 'month']
#features = [col for col in train_df.columns if col not in exclude_col]   

features = ['sunshine', 'cloud', 'humidity', 'pressure', 'dewpoint', 'windspeed']      


# This function performs feature engineering by creating new interaction, temporal,
# and transformed features to enrich the dataset for predictive modeling.   

def feature_engineering(df):
    df['cloud_humidity'] = df['cloud'] * df['humidity']
    df['pressure_humidity'] = df['pressure'] * df['humidity']
    df['sunshine_temperature'] = df['sunshine'] * df['temperature']
    df['humidity_sunshine'] = df['humidity'] * df['sunshine']
    df['cloud_wind'] = df['cloud'] * df['windspeed']
    df['inverse_humidity'] = 100 - df['humidity']
    
    # Ratio              
    df['humidity_pressure_ratio'] = df['humidity'] / df['pressure']
    df['cloud_sunshine_ratio'] = df['cloud'] / (df['sunshine'] + 1)
    df['cloud_humidity_ratio'] = df['cloud'] / (df['humidity'] + 1)

    
    # May detect `Rain-Bearing Winds`
    # Positve value ->> wind blows from west to east 
    # Negative value ->> wind blows from east to west 
    df['wind_west_east'] = df['windspeed'] * np.cos(np.radians(df['winddirection']))
    
    # Positive value ->> wind blows from south to north
    # Negative value ->> wind blows from north to south 
    df['wind_south_north'] = df['windspeed'] * np.sin(np.radians(df['winddirection']))

    # Wind magnitude 
    df['wind_magnitude'] = np.sqrt(df['wind_west_east']** 2 + df['wind_south_north']**2) 
    
    # Range
    df['temp_range'] = df['maxtemp'] - df['mintemp']
    # `pressure_dewpoint_diff` might help detect Fog/Mist 
    df['pressure_dewpoint_diff'] = df['pressure'] - df['dewpoint'] 
    # Might indicate how close air is to saturation
    df['temperature_dewpoint_diff'] = df['temperature'] - df['dewpoint']

    # Cycling encoding for `day` and `month`
    df['day_sin'] = np.sin(2 * np.pi * df['day'] / 365)
    df['day_cos'] = np.cos(2 * np.pi * df['day'] / 365)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    # Relative features 
    for feature in features:
        df[f'{feature}_relative'] = df[feature] / df.groupby('month')[feature].transform('max') 
        
    # Lag features 
    for feature in features:
        for lag in [1, 2, 3]:
            df[f'{feature}_lag{lag}'] = df[feature].shift(lag).fillna(df[feature].median()) 

    # Rolling Median
    for feature in features:
        df[f'{feature}_rollings3'] = df[feature].rolling(3, min_periods=1).median()
      
    return df      

train_df = feature_engineering(train_df)
test_df = feature_engineering(test_df)


print(f'Now there are total {train_df.shape[1]} features')    
train_df.columns       


# Features & Target   
X = train_df.drop(columns=['id', 'rainfall'])
y = train_df['rainfall']
X_test = test_df.drop(columns=['id'])

# Parameters
params = {'objective': 'binary',
          'metric': 'auc',
          'boosting_type': 'gbdt',
          'learning_rate': 0.02,
          'num_leaves': 31,
          'max_depth': 4,
          'min_child_sample': 75,
          'feature_fraction': 0.9,
          'bagging_fraction': 0.9,
          'bagging_freq': 5,
          'lambda_l1': 5,
          'lambda_l2': 5,
          'n_estimators': 1000,
          'device': 'gpu'  
}  

cat_params = {
    'iterations': 1500,         
    'learning_rate': 0.004,     
    'depth': 6,                
    'l2_leaf_reg': 12,         
    'border_count': 40,        
    'min_data_in_leaf': 75, 
    'bootstrap_type': 'Bayesian',
    'task_type': 'GPU'       
}     


feature_importances = np.zeros(X.shape[1])       

# shap_importances = np.zeros(X.shape[1])

n_splits = 5 
tscv = TimeSeriesSplit(n_splits=n_splits)  

auc_scores = []

#test_preds = np.zeros(len(X_test))

print("### Training LightGBM Model with 15 Features ###")    
for fold, (train_idx, valid_idx) in enumerate(tscv.split(X)):
    print(f"\n### Training Fold {fold+1} ###")
    
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_valid, y_valid = X.iloc[valid_idx], y.iloc[valid_idx]
    
    model = LGBMClassifier(**params, verbose=-1)
    
    model.fit(X_train, y_train, 
              eval_set=[(X_valid, y_valid)], 
              callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)])

    # Predict & Evaluate 
    y_pred = model.predict_proba(X_valid)[:, 1]
    auc = roc_auc_score(y_valid, y_pred)
    auc_scores.append(auc)
    print(f'Fold {fold+1} AUC-ROC: {auc:.6f}')

    # Predict on the Test set (Averaging over folds)
    #test_preds += model.predict_proba(X_test)[:, 1] / tscv.n_splits

    feature_importances += model.booster_.feature_importance(importance_type='gain') / tscv.n_splits 

    # Compute SHAP values for the validation set   
    # explainer = shap.TreeExplainer(model)
    # shap_values = explainer.shap_values(X_valid)

    # SHAP values are always list before indexing
    # if isinstance(shap_values, list):
        # shap_values = shap_values[1]         
    
    # Accumulate SHAP importances
    # shap_importances += np.abs(shap_values).mean(axis=0) / n_splits

# Final AUC Score
print(f'\nAverage AUC-ROC: {np.mean(auc_scores):.6f}')   

# Create feature importance dataframe
importance_df = pd.DataFrame({'Feature': X.columns, 
                              'Importances': feature_importances}) 

# Sort by importance and select top 15 features 
top = importance_df.sort_values(by='Importances', ascending=False).head(15)['Feature'].tolist()        


print(f'Now there are {len(top)} features')
top


# Filter the dataset with top 15 features
X_selected = X[top]
X_test_selected = X_test[top]

n_splits = 5 
tscv = TimeSeriesSplit(n_splits=n_splits)  

auc_scores = []

# test_preds = np.zeros(len(X_test_selected))  

print("### Training LightGBM Model with 15 Features ###")
for fold, (train_idx, valid_idx) in enumerate(tscv.split(X_selected)):    
    print(f"\n### Training Fold {fold+1} ###")
    
    X_train, y_train = X_selected.iloc[train_idx], y.iloc[train_idx]
    X_valid, y_valid = X_selected.iloc[valid_idx], y.iloc[valid_idx]
    
    model = LGBMClassifier(**params, verbose=-1)
    
    model.fit(X_train, y_train, 
              eval_set=[(X_valid, y_valid)], 
              callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)])

    # Predict & Evaluate 
    y_pred = model.predict_proba(X_valid)[:, 1]
    auc = roc_auc_score(y_valid, y_pred)
    auc_scores.append(auc)
    print(f'Fold {fold+1} AUC-ROC: {auc:.6f}')

    # Predict on the Test set (Averaging over folds)
    # test_preds += model.predict_proba(X_test_selected)[:, 1] / tscv.n_splits

# Final AUC Score
print(f'\nLightGBM Average AUC-ROC: {np.mean(auc_scores):.6f}') 


# Number of folds      
n_splits = 5  
tscv = TimeSeriesSplit(n_splits=n_splits)

# To store AUC scores for each fold
auc_scores = []  

# test_preds = np.zeros(len(X_test_selected))  
print("### Training CatBoost Model with 15 Features ###")
for fold, (train_idx, valid_idx) in enumerate(tscv.split(X_selected)):
    print(f"\n### Training Fold {fold+1} ###")
    
    # Split training and validation data for the current fold
    X_train, y_train = X_selected.iloc[train_idx], y.iloc[train_idx]
    X_valid, y_valid = X_selected.iloc[valid_idx], y.iloc[valid_idx]
    
    # Initialize CatBoostClassifier
    model = CatBoostClassifier(**cat_params, verbose=0)  
    
    # Fit the model with training data
    model.fit(X_train, y_train, 
              eval_set=[(X_valid, y_valid)], 
              early_stopping_rounds=50)
    
    # Predict probabilities for the validation set
    y_pred = model.predict_proba(X_valid)[:, 1]  
    
    # Calculate AUC-ROC for the fold
    auc = roc_auc_score(y_valid, y_pred)
    auc_scores.append(auc)
    print(f'Fold {fold+1} AUC-ROC: {auc:.6f}')
    
    # Predict probabilities for the test set (average over folds)
    # test_preds += model.predict_proba(X_test_selected)[:, 1] / n_splits

# Final AUC Score across all folds
print(f'\nCatBoost Average AUC-ROC: {np.mean(auc_scores):.6f}')  


# Train on the entire training dataset 
final_model = CatBoostClassifier(**cat_params, verbose=0)
final_model.fit(X_selected, y)     

# Predict on the full training dataset 
y_pred = final_model.predict_proba(X_selected)[:, 1]
auc = roc_auc_score(y, y_pred)
print(f'\nFinal Model AUC-ROC on Full Train Data: {auc:.6f}')   


# Predict on the test set
test_preds = final_model.predict_proba(X_test_selected)[:, 1]   


submission = pd.DataFrame({
    'id': test_df['id'],
    'rainfall': test_preds
})

submission.to_csv('submission.csv', index=False)
print('Final submission file created')

submission.head()    


# Descriptive Statistics for train and test predictions
train_predictions = pd.Series(y_pred).describe() 
test_predictions = submission['rainfall'].describe()  

# Creating a summary DataFrame
summary_df = pd.DataFrame({
    'train_predictions': train_predictions,
    'test_predictions': test_predictions
})

# Histogram  
plt.figure(figsize=(8, 3)) 
plt.hist(y_pred, bins=50, alpha=0.6, label='Train Predictions', color='blue') 
plt.hist(submission['rainfall'], bins=50, alpha=0.6, label='Test Predictions', color='orange') 
plt.xlabel('Predicted Probability') 
plt.ylabel('Frequency') 
plt.title('Train vs Test Predictions Distribution', weight='bold') 
plt.legend();

summary_df      

