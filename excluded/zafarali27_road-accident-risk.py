#system handling
import os
import time
import warnings
warnings.filterwarnings('ignore')

#data handling
import numpy as np # linear algebra
import pandas as pd # data processing, 
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


import lightgbm as lgb
from lightgbm import LGBMRegressor
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split,KFold, StratifiedKFold,cross_val_score
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, LabelEncoder


train_df = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e10/sample_submission.csv')


from colorama import Fore, Style

# Print the shape of the dataframe (number of rows and columns)
print(Fore.CYAN + "train_df shape: " + Style.RESET_ALL)
print(f"{train_df.shape}\n")

# Print basic information about the dataframe (column names, data types, non-null values)
print(Fore.GREEN + "train_df info: " + Style.RESET_ALL)
print(f"{train_df.info()}\n") 

# Print the count of missing (NaN) values in each column
print(Fore.YELLOW + "train_df isnull sum: " + Style.RESET_ALL)
print(f"{train_df.isnull().sum()}\n")

# Print summary statistics for numerical columns (count, mean, std, min, max, etc.)
print(Fore.MAGENTA + "train_df describe: " + Style.RESET_ALL)
print(f"{train_df.describe()}\n")


# define the numerical and categorical columns
numerical_cols = train_df.select_dtypes(include=["int64","float64"]).columns
categorical_cols = train_df.select_dtypes(include=["object","bool"]).columns

print(f" We have features: {len(numerical_cols)} numerical features {numerical_cols}")
print("-"*150)
print(f" We have features: {len(categorical_cols)} categorical features {categorical_cols}")


# Define color palette
palette = sns.color_palette("husl", len(numerical_cols))

#to show Distribution
plt.figure(figsize=(25, 15))
for i, col in enumerate(numerical_cols, 1):
    plt.subplot(3, 3, i)
    sns.histplot(train_df[col], kde=True, color=palette[i-1], bins=30)
    plt.title(f'Distribution of {col}')
plt.tight_layout()
plt.show()


for col in categorical_cols:
    count = train_df[col].value_counts()
    plt.figure(figsize=(18,6))
    plt.subplot(1,2,1)

    sns.countplot(data = train_df, x = col, palette="inferno", order = count.index)
    plt.title(f"Countplot of {col}")
    plt.xlabel(col)
    plt.ylabel("Frquency")

    plt.subplot(1,2,2)
    plt.pie(count, labels = count.index, autopct='%1.1f%%', startangle=140)
    plt.title(f"Pie chart of {col}")
    plt.tight_layout()
    plt.show()


correlation_matrix = train_df[numerical_cols].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix of Numerical Features")
plt.show()


print("\n" + "="*50)
print("ðŸ§® FEATURE ENGINEERING")
print("="*50)

def create_new_features(data):

    df = data.copy()  # work on a copy

    # Lane density: number of lanes divided by speed limit
    df['lane_density'] = df['num_lanes'] / (df['speed_limit'] + 1e-5)
    
    # Curvature per lane
    df['curvature_per_lane'] = df['curvature'] / (df['num_lanes'] + 1e-5)
    
    # Night and poor lighting
    df['night_poor_lighting'] = ((df['time_of_day'] == 'Night') & (df['lighting'] == 'Poor')).astype(int)
    
    # School season and school zone sign present
    df['school_season_zone'] = ((df['school_season'] == 1) & (df['road_signs_present'].astype(str).str.contains('School', na=False))).astype(int)
    
    # Holiday or weekend (assuming time_of_day can be a day name)
    df['holiday_or_weekend'] = ((df['holiday'] == 1) | (df['time_of_day'].isin(['Saturday', 'Sunday']))).astype(int)
    
    # Accidents per lane
    df['accidents_per_lane'] = df['num_reported_accidents'] / (df['num_lanes'] + 1e-5)
    
    # Weather and lighting interaction
    df['weather_lighting'] = df['weather'].astype(str) + '_' + df['lighting'].astype(str)
    
    return df

# Apply to both train and test
train_features = create_new_features(train_df)
test_features = create_new_features(test_df)

# Get new feature columns
new_features = [col for col in train_features.columns if col not in train_df.columns]
print(f"âœ¨ Created {len(new_features)} new features:")

for feature in new_features:
    print(f" .{feature}")


from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

def encode_categorical(df):
    """
    Encode all categorical (object/bool) columns in the dataframe using LabelEncoder.
    """
    # Select categorical columns
    categorical_cols = df.select_dtypes(include=["object", "bool"]).columns
    encoder = LabelEncoder()
    # Iterate through selected columns and encode them
    for col in categorical_cols:
        df[col] = encoder.fit_transform(df[col].astype(str))
    return df

# Encode categorical features in train and test dataframes
train_features = encode_categorical(train_features)
test_features = encode_categorical(test_features)


X = train_features.drop(columns=[ 'accident_risk'])
# Assign the target variable
y = train_features['accident_risk']

# Scale features using StandardScaler
scaler = StandardScaler()
# Fit the scaler on the training data and transform both training and testing data
X_train_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(test_features)

# Assign scaled features to X_train and X_test
X_train = X_train_scaled
X_test = X_test_scaled
# Convert the target variable to a numpy array
y_train = np.array(y)


# cross-validation setup
N_splits = 5
kfold = KFold(n_splits = N_splits, shuffle = True , random_state = 42)


# Storage for predictions 
off_XGB = np.zeros(len(X_train))
off_LGB = np.zeros(len(X_train))
off_CAT = np.zeros(len(X_train))

test_XGB = np.zeros(len(X_test))
test_LGB = np.zeros(len(X_test))
test_CAT = np.zeros(len(X_test))

# Store trained models & scores
models_XGB, models_LGB, models_CAT = [], [], []
scores_XGB, scores_LGB, scores_CAT = [], [], []


XGB_params = {
    'n_estimators': 2000,
    'learning_rate': 0.02,
    'max_depth': 8,
    'min_child_weight': 2,
    'subsample': 0.75,
    'colsample_bytree': 0.75,
    'reg_alpha': 0.5,
    'reg_lambda': 1.0,
    'random_state': 42,
    'tree_method': 'gpu_hist',
    'gpu_id': 0,
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse'
}

print("Training XBGBoost with acceleration")
print(">" * 60)

for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X_train), 1):
    print(f"Fold {fold_idx}/{N_splits}", end=" ")

    X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
    y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]

    model = xgb.XGBRegressor(**XGB_params)
    model.fit(
        X_fold_train, y_fold_train,
        eval_set = [(X_fold_val, y_fold_val)],
        early_stopping_rounds = 100,
        verbose = False
    )

    off_XGB[val_idx] = model.predict(X_fold_val)
    test_XGB += model.predict(X_test) / N_splits

    fold_rmse = np.sqrt(mean_squared_error(y_fold_val, off_XGB[val_idx]))
    scores_XGB.append(fold_rmse)
    models_XGB.append(model)

    print(f"RMSE: {fold_rmse:.6f} | Best iter: {model.best_iteration}")
    xgb_off_rmse = np.sqrt(mean_squared_error(y_train, off_XGB))
    print(f"CV Std: {np.std(scores_XGB):.6f}")
    print("=" * 60)


#  second model LightGBM 

LGB_params = {
    'n_estimators': 2000,
    'learning_rate': 0.02,
    'num_leaves': 63,
    'max_depth': 8,
    'min_child_samples': 20,
    'subsample': 0.75,
    'colsample_bytree': 0.75,
    'reg_alpha': 0.5,
    'reg_lambda': 1.0,
    'random_state': 42,
    'device': 'gpu',
    'gpu_platform_id': 0,
    'gpu_device_id': 0,
    'verbose': -1
}

print("Training LightGBM with GPU acceleration")
print(">" * 60)

for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X_train), 1):
    print(f"Fold {fold_idx}/{N_splits}",end="")

    X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
    y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]

    model =  lgb.LGBMRegressor(**LGB_params)
    model.fit(
        X_fold_train, y_fold_train,
        eval_set = [(X_fold_val, y_fold_val)],
        # callbacks = [lgb.early_stopping(100), lgb.lgb_evaluation(0)]
        callbacks = [lgb.early_stopping(100), lgb.log_evaluation(0)]
    )

    off_LGB[val_idx] = model.predict(X_fold_val)
    test_LGB += model.predict(X_test) / N_splits
    # test_LGB += model.predict(X_test) / N_splits


    fold_rmse = np.sqrt(mean_squared_error(y_fold_val, off_LGB[val_idx]))
    scores_LGB.append(fold_rmse)
    models_LGB.append(model)

    print(f"RMSE: {fold_rmse:.6f} | Best iter: {model.best_iteration_}")

LGB_OFF_RMSE = np.sqrt(mean_squared_error(y_train, off_LGB))

print(f"\nLightGBM OFF RMSE: {LGB_OFF_RMSE:.6f}")
print(f"CV Std: {np.std(scores_LGB):.6}")
print(">" * 60)    


# third model Catboost
CAT_params = {
    'iterations':1500,
    'learning_rate' :0.05,
    'depth' : 6,
    'l2_leaf_reg' : 3,
    'loss_function' :'RMSE',
    'eval_metric':'RMSE',
    'random_seed':42,
    'early_stopping_rounds':50,
    'verbose':100,
    "task_type":"GPU"
}

print("Traning CATBoost with GUP acceleration")
print(">" * 60)



for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X_train), 1):
    print(f"Fold {fold_idx}/{N_splits}",end="")

    X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
    y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]

    model = CatBoostRegressor(**CAT_params)
    model.fit(
        X_fold_train, y_fold_train,
        eval_set = [(X_fold_val, y_fold_val)]
    )

    off_CAT[val_idx] = model.predict(X_fold_val)
    test_CAT += model.predict(X_test) / N_splits

    fold_rmse = np.sqrt(mean_squared_error(y_fold_val, off_CAT[val_idx]))
    scores_CAT.append(fold_rmse)
    models_CAT.append(model)

    print(f"RMSE: {fold_rmse:.6f} | Best iter: {model.best_iteration_}")

cat_off_rmse = np.sqrt(mean_squared_error(y_train, off_CAT))

print(f"\nCATBoost off RMSE: {cat_off_rmse:.6f}")
print(f"CV Std: {np.std(scores_CAT):.6f}") 
print(">" * 60)


# Weighted Ensemble
w_xgb = 0.6
w_lgb = 0.4

ensemble_oof = w_xgb *  off_XGB  + w_lgb * off_LGB
ensemble_test = w_xgb * test_XGB + w_lgb * test_LGB

# Evaluate OOF RMSE
ens_oof_rmse = np.sqrt(mean_squared_error(y_train, ensemble_oof))
print(f"Weighted Ensemble OOF RMSE: {ens_oof_rmse:.6f}")
print("=" * 60)

# ==========================
# Create submission
# ==========================
submission = pd.DataFrame({
    'id': test_df['id'],
    'accident_risk': ensemble_test
})


# Save file
submission.to_csv('Submission.csv', index=False)

print("Submission Created Successfully")

