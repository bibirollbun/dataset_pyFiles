!pip install feature-engine


import numpy as np 
import pandas as pd 
import time
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
warnings.simplefilter('ignore')

from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline

from feature_engine.imputation import MeanMedianImputer, CategoricalImputer
from feature_engine.encoding import (
    OneHotEncoder, 
    MeanEncoder,
    OrdinalEncoder
)


from xgboost import XGBRegressor

import optuna


train_data = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv', index_col='id')
train_data_extra = pd.read_csv('/kaggle/input/playground-series-s5e2/training_extra.csv', index_col='id')
test_data = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv', index_col='id')


print("Training data shape: ", train_data.shape)
print("Training Extra data shape: ", train_data_extra.shape)
print("Testing data shape: ", test_data.shape)


print('Training columns: ',train_data.columns, "\n")
print('Training Extra columns: ',train_data_extra.columns, "\n")
print('Testing columns: ',test_data.columns)


train_data.head()


train_data_extra.head()


merged_data = pd.concat([train_data, train_data_extra], axis=0, ignore_index=False)

print('Merged data shape: ', merged_data.shape)


for var in merged_data.columns:
    print('Column: ',var)
    print('Number of unique labels: ',merged_data[var].nunique(), '\n')


print(merged_data.dtypes)


merged_data['Compartments'] = merged_data['Compartments'].astype('int64')
columns_to_analyze = merged_data.columns.drop('Price')

categorical_columns = [col for col in columns_to_analyze if merged_data[col].dtype == 'object']
discrete_columns = [col for col in columns_to_analyze if merged_data[col].dtype in ['int64']]
continuous_columns = [col for col in columns_to_analyze if merged_data[col].dtype in ['float64']]

print("There are {}".format(len(categorical_columns)), 'categorical columns:',categorical_columns)
print("There is {}".format(len(discrete_columns)), 'discrete column:', discrete_columns)
print("There is {}".format(len(continuous_columns)), 'continuous column:', continuous_columns)


merged_data.isnull().mean()[merged_data.isnull().mean() > 0]


test_data.isnull().mean()[test_data.isnull().mean() > 0]


for var in categorical_columns:
    print(merged_data[var].value_counts())
    print()
    


def diagnostic_plots(df, target_column):

    plt.figure(figsize=(15, 5))
    
    # Histogram
    plt.subplot(1, 3, 1)
    sns.histplot(df[target_column], color='skyblue', bins=30)
    plt.title('Histogram')
    plt.xlabel(target_column)
    
    # Boxplot
    plt.subplot(1, 3, 2)
    sns.boxplot(y=df[target_column], color='lightcoral')
    plt.title('Boxplot')
    
    # QQ Plot (normality check)
    plt.subplot(1, 3, 3)
    stats.probplot(df[target_column], dist="norm", plot=plt)
    plt.title('QQ Plot')
    
    plt.tight_layout()
    plt.show()



diagnostic_plots(merged_data, target_column='Compartments')


merged_data.describe()


# Create copies of the dataframes without modifying the originals
merged_data_for_imputation = merged_data.drop('Price', axis=1)
test_data_for_imputation = test_data.copy()

imputation_pipeline = Pipeline([
    ('categorical_imputer', CategoricalImputer(variables=categorical_columns, imputation_method='frequent')),
    ('numerical_imputer', MeanMedianImputer(variables=continuous_columns, imputation_method='median'))
])

merged_data_imputed = imputation_pipeline.fit_transform(merged_data_for_imputation)

test_data_imputed = imputation_pipeline.transform(test_data_for_imputation)

# Add back the Price column to merged_data_imputed
merged_data_imputed['Price'] = merged_data['Price']

print("Missing values in merged_data after imputation:")
print(merged_data_imputed.isnull().sum()[merged_data_imputed.isnull().sum() > 0])
print("\nMissing values in test_data after imputation:")
print(test_data_imputed.isnull().sum()[test_data_imputed.isnull().sum() > 0])


onehot_vars = ['Laptop Compartment', 'Waterproof', 'Style', 'Color']
mean_vars = ['Brand', 'Material']
ordinal_vars = ['Size']

# Create copies without the Price column for encoding
merged_data_for_encoding = merged_data_imputed.drop('Price', axis=1)
test_data_for_encoding = test_data_imputed.copy()

y = merged_data_imputed['Price']

# Create encoding pipeline
encoding_pipeline = Pipeline([
    ('ordinal', OrdinalEncoder(
        encoding_method='ordered',
        variables=ordinal_vars,
    )),
    ('mean', MeanEncoder(
        variables=mean_vars,
        ignore_format=True
    )),
    ('onehot', OneHotEncoder(
        variables=onehot_vars,
        drop_last=True
    ))
])

merged_data_clean = encoding_pipeline.fit_transform(merged_data_for_encoding, y)

test_data_clean = encoding_pipeline.transform(test_data_for_encoding)

merged_data_clean['Price'] = y


print("\nNew columns created:")
print(merged_data_clean.columns.tolist())


scaler = RobustScaler()

numeric_features = ['Compartments', 'Weight Capacity (kg)'] + \
                  [col for col in merged_data_clean.columns 
                   if col not in ['Compartments', 'Weight Capacity (kg)', 'Price']]

price = merged_data_clean['Price']

merged_data_clean[numeric_features] = scaler.fit_transform(merged_data_clean[numeric_features])

test_data_clean[numeric_features] = scaler.transform(test_data_clean[numeric_features])

merged_data_clean['Price'] = price


# Verify the scaling
print("Sample means after scaling:")
print(merged_data_clean[numeric_features].mean().head())
print("\nSample standard deviations after scaling:")
print(merged_data_clean[numeric_features].std().head())


from sklearn.model_selection import KFold

def objective(trial):
    params = {
        "max_depth": trial.suggest_int("max_depth", 3, 15),          
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),  
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 7),  
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),    
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0), 
        "n_estimators": trial.suggest_int("n_estimators", 100, 3000), 
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),  
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True), 
        "gamma": trial.suggest_float("gamma", 1e-8, 1.0, log=True)  
    }

    model = XGBRegressor(
        tree_method="gpu_hist",
        random_state=42,
        **params
    )

    X = merged_data_clean.drop(columns=['Price'])
    y = merged_data_clean['Price']

    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    
    rmse_scores = []
    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        rmse_scores.append(rmse)
    
    return np.mean(rmse_scores)


study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=20)

best_params = study.best_trial.params
best_cv_rmse = study.best_value


print("Best parameters:", best_params)
print("Best RMSE:", best_cv_rmse)


model = XGBRegressor(
    tree_method = "gpu_hist",
    random_state=42,
    **best_params
)


X_train = merged_data_clean.drop(columns=['Price'])
y_train = merged_data_clean['Price']

model.fit(
    X_train, 
    y_train,
    eval_set=[(X_train, y_train)],
    eval_metric='rmse',
    verbose=False,
)


feature_names = merged_data_clean.drop(columns=['Price']).columns
importance = model.feature_importances_

sorted_idx = np.argsort(importance)[::-1]

plt.figure(figsize=(12, 8))
plt.barh(range(len(sorted_idx)), importance[sorted_idx])

plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx])

plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.title("XGBoost Feature Importance")

plt.tight_layout()
plt.show()


preds = model.predict(test_data_clean)


submission = pd.read_csv('/kaggle/input/playground-series-s5e2/sample_submission.csv')
submission = pd.DataFrame({'id': submission.id, 'Price': preds})
submission.to_csv('/kaggle/working/submission.csv', index=False)
display(submission)

