# Importing Libraries
# First, we import all necessary Python libraries for data processing, visualization, modeling, and evaluation

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as plt
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')



train = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e4/sample_submission.csv')

# Save real test ids
test_real_id = test['id']


train.isnull().sum()


test.isnull().sum()


# For train
train['Episode_Length_minutes'] = train['Episode_Length_minutes'].fillna(train['Episode_Length_minutes'].mean())
train['Guest_Popularity_percentage'] = train['Guest_Popularity_percentage'].fillna(train['Guest_Popularity_percentage'].mean())
train['Number_of_Ads'] = train['Number_of_Ads'].fillna(train['Number_of_Ads'].mean())

#remove column with name id from train
train.drop(columns=['id'], inplace = True ,errors = 'ignore')

# For test
test['Episode_Length_minutes'] = test['Episode_Length_minutes'].fillna(test['Episode_Length_minutes'].mean())
test['Guest_Popularity_percentage'] = test['Guest_Popularity_percentage'].fillna(test['Guest_Popularity_percentage'].mean())

#remove column with name id from test
# test.drop(columns=['id'], inplace = True ,errors = 'ignore')


# Automatically detect columns
number_columns = train.select_dtypes(include=(['int64', 'float64'])).columns.tolist()
catogorical_columns = train.select_dtypes(include=(['object'])).columns.tolist()



# Check missing values in numerical columns
train[number_columns].isnull().mean().sort_values(ascending=False)

# # Basic stats
train[number_columns].describe()


# Visualize distributions

for col in number_columns:
    plt.figure(figsize=(6, 4))
    sns.histplot(train[col], kde=True)
    plt.title(f"Distribution of {col}")
    plt.show()



# Check missing values in categorical columns
train[catogorical_columns].isnull().mean().sort_values(ascending=False)

# # Unique values and top categories
for col in catogorical_columns:
    print(f"\nColumn: {col}")
    print(f"Unique values: {train[col].nunique()}")
    print(train[col].value_counts().head(5))



for col in number_columns:
    plt.figure(figsize=(6, 4))
    sns.histplot(train[col], kde=True)
    plt.title(f"Distribution of {col}")
    plt.show()


# Encode them
le = LabelEncoder()
for col in catogorical_columns:
    train[col] = le.fit_transform(train[col])
    test[col] = le.transform(test[col])



# Prepare train and test 
X = train.drop(['Listening_Time_minutes'], axis=1)
y = train['Listening_Time_minutes']
X_test = test.drop(['id'], axis=1)



# Label Encode categorical columns
label_encoders = {}

for col in X.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    X_test[col] = le.transform(X_test[col])  # Important: transform test data too
    label_encoders[col] = le



# Create LightGBM Model
# 5-Fold CV
kf = KFold(n_splits=5, shuffle=True, random_state=42)
val_scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
    print(f"ðŸ”µ Fold {fold}")
    
    X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
    y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
    
    # Create dataset for LightGBM
    train_data = lgb.Dataset(X_train_fold, label=y_train_fold)
    val_data = lgb.Dataset(X_val_fold, label=y_val_fold)
    
    # Specify parameters
    params = {
        'objective': 'regression',
        'metric': 'l2',  # For RMSE
        'learning_rate': 0.05,
        'max_depth': 7,
        'n_estimators': 1000
    }
    
    # Train the model with early stopping
    model = lgb.train(
        params,
        train_data,
        valid_sets=[val_data],
        # early_stopping_rounds=50,  # Stop if no improvement after 50 rounds
        # verbose_eval=100
    )
    
    # Predict and calculate RMSE
    y_val_pred = model.predict(X_val_fold)
    fold_rmse = mean_squared_error(y_val_fold, y_val_pred) ** 0.5
    val_scores.append(fold_rmse)
    print(f"âœ… Fold {fold} RMSE: {fold_rmse:.4f}\n")

# Calculate average RMSE
print(f"ðŸ“ˆ Average CV RMSE: {np.mean(val_scores):.4f}")




final_model = LGBMRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=7,
    random_state=42
)

final_model.fit(X, y)
print("âœ… Final model trained on full data!")



# Predict on test data
test_predictions = final_model.predict(X_test)



submission = pd.DataFrame({
    'id': test_real_id,  # Use real id
    'Listening_Time_minutes': test_predictions
})

submission.to_csv('submission.csv', index=False)
print("âœ… Submission file 'submission.csv' saved!")


