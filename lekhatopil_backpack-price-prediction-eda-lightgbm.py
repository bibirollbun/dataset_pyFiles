# Import packages              
import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
import missingno as msno

import optuna
from xgboost import XGBRegressor  
from catboost import CatBoostRegressor
import lightgbm as lgb
from lightgbm import LGBMRegressor, early_stopping
from sklearn.model_selection import train_test_split, cross_val_score, KFold 
from sklearn.metrics import mean_squared_error, make_scorer


import warnings 
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)  
warnings.filterwarnings("ignore", category=UserWarning)
warnings.simplefilter("ignore", category=RuntimeWarning)


# Read Train and Test data   
train_df = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')

train_df.head()


test_df.head()


# Check Train and Test data size  
print('Train:', train_df.shape)
print('Test:', test_df.shape)      


# Display train_df  
train_df.head()


# Display test_df
test_df.head()  


# Change column names to lower_case & replace space with underscore
train_df.columns = train_df.columns.str.replace(' ', '_').str.lower()
test_df.columns = test_df.columns.str.replace(' ', '_').str.lower()

print('Train:\n', train_df.columns)
print('Test:\n',  test_df.columns) 


# Check for Duplicates
print('Count of Duplicated rows in train_df:', train_df.duplicated().sum())
print('Count of Duplicated rows in test_df:', test_df.duplicated().sum())


# Check whether both datasets train_df and test_df have same categorical columns
train_df.select_dtypes(include=['object']).columns.tolist() == test_df.select_dtypes(include=['object']).columns.tolist()


# Check whether both datasets have same unique categories for categorical variables
# Extract all catgorical variables
categorical = train_df.select_dtypes(include=['object']).columns.tolist()

# Creat dictionary to hold unqiue categories for each categorical variable for train and test datasets
categorical_train = {var: set(train_df[var].dropna().unique()) for var in categorical}
categorical_test = {var: set(test_df[var].dropna().unique()) for var in categorical}

categorical_train == categorical_test 


# Check categorical variables with unique categories in train_df
print('Train Dataset:')
categorical_train


# Check categorical variables with unique categories in test_df
print('Test Dataset:')
categorical_test


print('Train Dataset:')
train_df.info()
print()
print('Test Dataset')
test_df.info()


# Create subplots
fig, axes = plt.subplots(len(categorical), 2, figsize=(14, 4 * len(categorical)))

# Loop through each categorical variable
for i, var in enumerate(categorical):
    # Train DataFrame Pie Chart 
    train_df[var].value_counts().plot.pie(ax=axes[i, 0], autopct='%1.1f%%', 
                                         startangle=90, title=f'Train: {var.capitalize()}')
    # Test DataFrame Pie Chart
    test_df[var].value_counts().plot.pie(ax=axes[i, 1], autopct='%1.1f%%', 
                                         startangle=90, title=f'Test: {var.capitalize()}')   


# Extract all numerical variables               
numerical = train_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
numerical.remove('id')
numerical.remove('price')
numerical        


# Check for infinite values in the `price` column
train_df['price'].isin([np.inf, -np.inf]).any()      


# Setup subplots
fig, axes = plt.subplots(len(numerical), 2, figsize=(13, 5 * len(numerical)))

# Plot Histogram for Train & Test data
for i, var in enumerate(numerical):
    axes[i, 0].hist(train_df[var], alpha=0.5, label='Train')
    axes[i, 0].hist(test_df[var], alpha=0.5, label='Test')
    axes[i, 0].set_title(f'Histogram for {var}', weight='bold')
    axes[i, 0].legend()

    # Prepare data for boxplot
    combined = pd.concat([train_df[var].to_frame().assign(dataset='Train'),
                          test_df[var].to_frame().assign(dataset='Test')
                         ])
    # Plot Boxplot
    sns.boxplot(data=combined, x='dataset', y=var, ax=axes[i, 1], palette='Set2')
    axes[i, 1].set_title(f'Boxplot for {var}', weight='bold')

plt.tight_layout()
plt.show()


print('Train:')
print(train_df[['compartments', 'weight_capacity_(kg)']].describe())
print()
print('Test:')
print(train_df[['compartments', 'weight_capacity_(kg)']].describe())


# Setup subplots
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot Histogram for `price`
sns.histplot(x=train_df['price'], ax=axes[0], color='green')
axes[0].set_title('Histogram for Price', weight='bold')

# Plot Boxplot for `price`
sns.boxplot(y=train_df['price'], ax=axes[1], color='orange')
axes[1].set_title('Boxplot for Price', weight='bold');  


# Descriptive statistic for `price`
train_df['price'].describe()         


# Count `price` with and above 148
train_df[train_df['price']>=148].shape[0]  


# Create subplots for Boxplot  
fig, axs = plt.subplots(len(categorical) // 2 + 1, 2, figsize=(15, 24))

# Flatten the axes array to iterate over each subplot
axs = axs.flatten()

# Plot histogram for each variables based on price
for i, var in enumerate(categorical):
    ax = axs[i]
    sns.boxplot(data=train_df, x=var, y='price', palette='Set2', ax=ax)
    ax.set_title(f'Boxplot of {var} vs price', size=12, weight='bold')

# Hide extra subplots
for j in range(len(categorical), len(axs)):
    axs[j].axis('off')


train_df.groupby('color')['price'].describe()


# Missing values in train_df data 
train_count = train_df.isna().sum()
train_perct = round(train_df.isna().sum() / len(train_df) * 100, 2) 

# Missing values in test_df data 
test_count = test_df.isna().sum()
test_perct = round(test_df.isna().sum() / len(test_df) * 100, 2) 

# Create a DataFrame for missing value summary for both datasets
missing_summary = pd.DataFrame({
       'train_count': train_count,
       'train_perct': train_perct,
       'test_count' : test_count,
       'test_perct' : test_perct
})

missing_summary  


# Visualize missing data patterns
msno.matrix(train_df)
plt.title('Missing Data Pattern (Train Data)', size=18, weight='bold')
plt.show()

msno.matrix(test_df)
plt.title('Missing Data Pattern (Test Data)', size=18, weight='bold')
plt.show()

# Visualize missing data correlations
msno.heatmap(train_df, figsize=(8, 4))
plt.title('Missing Values Correlation (Train)', size=10, weight='bold')
plt.show()

msno.heatmap(test_df, figsize=(8, 4))
plt.title('Missing Values Correlation (Test)', size=10, weight='bold')
plt.show() 


# Mean imputation for missing values in weight_capacity_(kg) of train_df
train_df['weight_capacity_(kg)'].fillna(train_df['weight_capacity_(kg)'].mean(), inplace=True)

# Mean imputation for missing values in weight_capacity_(kg) of test_df
test_df['weight_capacity_(kg)'].fillna(test_df['weight_capacity_(kg)'].mean(), inplace=True)

# Verify imputation
print('Missing values in `weight_capacity_(kg)` for train_df:', train_df['weight_capacity_(kg)'].isna().sum())
print('Missing values in `weight_capacity_(kg)` for test_df:', test_df['weight_capacity_(kg)'].isna().sum())


# Create a `missing_cat` list to store categorical variables with missing values
missing_cat = [col for col in categorical if train_df[col].isna().sum().any()]
print('Categorical columns with missing values:\n', missing_cat)


# Define a function to impute missing values in categorical variables 
def impute_missing_categorical(train_df, test_df, missing_cat):
    for col in missing_cat:
        if train_df[col].isna().sum() > 0:
            train_df.loc[train_df[col].isna(), col] = np.random.choice(
                train_df[col].dropna(), size=train_df[col].isna().sum(), replace=True
            )
        if test_df[col].isna().sum() > 0:
            test_df.loc[test_df[col].isna(), col] = np.random.choice(
                test_df[col].dropna(), size=test_df[col].isna().sum(), replace=True
            ) 
    return train_df, test_df

train_df, test_df = impute_missing_categorical(train_df, test_df, missing_cat)

# Verify imputation
print('Missing values in Train:', train_df.isna().sum().sum())
print('Missing values in Test:', test_df.isna().sum().sum())  


def heatmap(df, df_name):
    plt.figure(figsize=(12, 5))
    sns.heatmap(df.drop(columns=['id']).corr(method='pearson', numeric_only=True), 
                annot=True, cmap='coolwarm')
    plt.title(f'Correlation Heatmap for {df_name}', fontsize=12, weight='bold')
    plt.show()
    
heatmap(train_df, 'Train Dataset')   
heatmap(test_df, 'Test Dataset')


def create_interaction_features(df):
    for i, col1 in enumerate(categorical):
        for col2 in categorical[i+1:]:
            df[f"{col1}_x_{col2}"] = df[col1] + "_" + df[col2] 

# Binary variables - (yes/no) --> convert to 0 and 1
# For laptop_compartment and waterproof
def create_binary(df):
    df['laptop_compartment'] = df['laptop_compartment'].map({'Yes': 1, 'No':0})
    df['waterproof'] = df['waterproof'].map({'Yes': 1, 'No':0})
    # Create laptop_waterproof - backpack with laptop compartment and waterproof
    # Brands might price higher for waterproof backpacks with laptop compartment 
    df['laptop_waterproof'] = ((df['laptop_compartment'] == 'Yes') & (
        df['waterproof'] == 'Yes')).astype(int)
    
    return df

train_df = create_binary(train_df)
test_df = create_binary(test_df)   





train_df.head(1)   


# Changing the datatype for categorical variables from object to category in train_df 
for col in train_df.select_dtypes('object').columns:
    train_df[col] = train_df[col].astype('category')

# Changing the datatype for categorical variables from object to category in test_df 
for col in test_df.select_dtypes('object').columns:
    test_df[col] = test_df[col].astype('category')     


# Number of folds
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# Create a new column initialized with NaNs
train_df['brand_material_avg'] = np.nan

# Perform KFold cross-validation
for train_idx, valid_idx in kf.split(train_df):
    train_fold, val_fold = train_df.iloc[train_idx], train_df.iloc[valid_idx]

    # Compute brand-material average price from training fold only
    brand_material_dict = train_fold.groupby(['brand', 'material'])['price'].mean().to_dict()

    # Assign to the validation fold
    train_df.loc[valid_idx, 'brand_material_avg'] = train_df.loc[valid_idx, ['brand', 'material']].apply(
        tuple, axis=1).map(brand_material_dict)

# Compute brand-material average price from the entire training set for the test set
brand_material_dict_full = train_df.groupby(['brand', 'material'])['price'].mean().to_dict()
test_df['brand_material_avg'] = test_df[['brand', 'material']].apply(tuple, axis=1).map(brand_material_dict_full)

# Verify if any missing values exist 
print("Missing values in train_df for `brand_material_avg:", train_df['brand_material_avg'].isna().sum())
print("Missing values in test_df for `brand_material_avg:", test_df['brand_material_avg'].isna().sum())       


# Set features and target
X_train = train_df.drop(columns=['id', 'price'])   
X_test = test_df.drop(columns=['id'])
y_train = train_df['price']  


# Function to compute RMSE  
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred)) 

# Define the best hyperparameters 
best_params = {'n_estimators': 13000, 
               'learning_rate': 0.0068, 
               'num_leaves': 270, 
               'max_depth': 2, 
               'min_child_samples': 400, 
               'subsample': 0.8, 
               'colsample_bytree': 0.75, 
               'reg_alpha': 41.46, 
               'reg_lambda': 70, 
               'cat_smooth': 60, 
               'min_gain_to_split': 2.0, 
               'max_bin': 140}

# Train LightGBM model 
oof_predictions = np.zeros(len(train_df))   
light_gbm_test_preds = np.zeros(len(test_df))

# Store per-fold RMSE
fold_rmse_list = []  

# Train LightGBM with KFold
for fold, (train_idx, valid_idx) in enumerate(kf.split(X_train)):  
    print(f"\n### Training Model - Fold {fold+1} ###")

    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[valid_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[valid_idx]

    model = lgb.LGBMRegressor(**best_params, verbose=-1)

    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        eval_metric='rmse',
        callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)]
    )

    # OOF Predictions
    val_preds = model.predict(X_val)
    oof_predictions[valid_idx] = val_preds

    # Calculate RMSE for each fold
    fold_rmse = rmse(y_val, val_preds)

    print(f'Fold {fold+1} RMSE: {fold_rmse:.6f}')  

    # Test Set Predictions (Averaged over folds)
    light_gbm_test_preds += model.predict(X_test) / n_splits 

# Compute final CV RMSE
cv_score = rmse(y_train, oof_predictions)
print(f"\n Overall CV RMSE (LightGBM): {cv_score:.6f}")     


# Plot Feature Importance (Split)
lgb.plot_importance(model.booster_, importance_type='split')
plt.title('Feature Importance by Split');        


# Plot Feature Importance (Gain)
lgb.plot_importance(model.booster_, importance_type='gain')
plt.title('Feature Importance by Gain');  


# Ensure test predictions are non-negative
test_predictions = np.maximum(light_gbm_test_preds, 0)   

# Create the submission DataFrame      
submission = pd.DataFrame({
    'id': test_df['id'],
    'Price': test_predictions
})

# Save the submission file
submission.to_csv('submission.csv', index=False)
print('Final submission file created')                     


submission.head()


submission['Price'].describe() 


submission['Price'].skew()            


plt.figure(figsize=(8, 4))
plt.hist(data=submission, x='Price', bins=100)
plt.title('Data Distribution of Test Prediction', size=12, weight='bold');

