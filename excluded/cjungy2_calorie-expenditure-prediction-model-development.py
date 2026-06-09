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


train = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
train.head(5)


train.info()


for col in train.columns:
    missing_rate = train[col].isnull().mean() * 100
    print(f"{col}: {missing_rate:.2f}%")


test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')
test.head()


test.info()


train.describe()


from sklearn.preprocessing import LabelEncoder

if 'Sex' in train.columns:
    le = LabelEncoder()
    train['Sex_encoded'] = le.fit_transform(train['Sex'])
    print(train[['Sex', 'Sex_encoded']].head())
else:
    pass


train['BMI'] = train['Weight'] / ((train['Height'] / 100) ** 2)
train.head()


import numpy as np

train['HRxDuration'] = train['Heart_Rate'] * train['Duration']

# Avoid division by zero if 'Body_Temp' contains zero values
train['Duration_Temp'] = train['Duration'] / train['Body_Temp'].replace(0, np.nan)

train.head()


cols = ['Height', 'Weight', 'Body_Temp', 'Heart_Rate']

for col in cols:
    Q1 = train[col].quantile(0.25)
    Q3 = train[col].quantile(0.75)
    train = train[(train[col] >= Q1) & (train[col] <= Q3)]


def calculate_bmr(row):
    if row['Sex_encoded'] == 1:  # Male (assuming 1 is male after encoding)
        return 66.47 + (13.75 * row['Weight']) + (5 * row['Height']) - (6.76 * row['Age'])
    else:  # Female (assuming 0 is female)
        return 655.1 + (9.56 * row['Weight']) + (1.85 * row['Height']) - (4.68 * row['Age'])

train['BMR'] = train.apply(calculate_bmr, axis=1)

print(train[['Sex_encoded', 'Age', 'Height', 'Weight', 'BMR']].head())


# Classify obesity by BMI
def classify_obesity(bmi):
    if bmi >= 30:
        return 'Obese'
    elif bmi >= 25:
        return 'Overweight'
    else:
        return 'Normal'

# Add column
train['Obesity_Status'] = train['BMI'].apply(classify_obesity)


def estimate_body_fat_percentage(row):
    age = row['Age']
    bmi = row['BMI']
    sex_encoded = row['Sex_encoded'] # 0 for Female, 1 for Male

    # Calculate body fat percentage
    if sex_encoded == 1: # Male
        return (1.20 * bmi) + (0.23 * age) - 16.2
    else: # Female
        return (1.20 * bmi) + (0.23 * age) - 5.4

train['Body_Fat_Percentage_Est'] = train.apply(estimate_body_fat_percentage, axis=1)

print(train[['BMI', 'Obesity_Status', 'Body_Fat_Percentage_Est']].head())


print("\nObesity Status Distribution:")
print(train['Obesity_Status'].value_counts())


## Correlation
# Only numerical columns

import numpy as np
df_numeric = train.select_dtypes(include=np.number)
c = df_numeric.corr().round(2)['Calories']
print(c.sort_values(ascending=False))


c = df_numeric.corr().round(2)['Calories']
print(c.sort_values(ascending=False))


import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_log_error
from sklearn.preprocessing import LabelEncoder

# Reload data
df = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')  # Training data
test_df = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')  # Test data

# Initialize LabelEncoder (initialized outside add_features function for consistency)
le = LabelEncoder()

# Feature Engineering
def add_features(df):
    df = df.copy()
    df['Sex_encoded'] = le.fit_transform(df['Sex'])

    df['BMI'] = df['Weight'] / ((df['Height'] / 100) ** 2)
    df['HRxDuration'] = df['Heart_Rate'] * df['Duration']
    # Handle cases where Body_Temp is 0 by replacing with the mean (more robust handling than np.nan)
    df['Duration_Temp'] = df['Duration'] / df['Body_Temp'].replace(0, df['Body_Temp'].mean())

    def calculate_bmr(row):
        if row['Sex_encoded'] == 1:  # Male
            return 66.47 + (13.75 * row['Weight']) + (5 * row['Height']) - (6.76 * row['Age'])
        else:  # Female
            return 655.1 + (9.56 * row['Weight']) + (1.85 * row['Height']) - (4.68 * row['Age'])
    df['BMR'] = df.apply(calculate_bmr, axis=1)

    def estimate_body_fat_percentage(row):
        age = row['Age']
        bmi = row['BMI']
        sex_encoded = row['Sex_encoded'] # 0 for Female, 1 for Male
        if sex_encoded == 1: # Male
            return (1.20 * bmi) + (0.23 * age) - 16.2
        else: # Female
            return (1.20 * bmi) + (0.23 * age) - 5.4
    df['Body_Fat_Percentage_Est'] = df.apply(estimate_body_fat_percentage, axis=1)

    # Handle cases where Duration is 0 (to avoid division by zero)
    df['HR_per_Minute'] = df['Heart_Rate'] / df['Duration'].replace(0, 1) # Replace 0 with 1 to prevent division by zero

    # Add new features: interaction and polynomial features
    df['Age_BMI_Interaction'] = df['Age'] * df['BMI']
    df['HeartRate_Temp_Interaction'] = df['Heart_Rate'] * df['Body_Temp']
    df['Duration_Squared'] = df['Duration'] ** 2
    df['Age_Squared'] = df['Age'] ** 2

    return df


# Apply derived features to training and test data
df = add_features(df)
test_df = add_features(test_df)


# Update list of features to use (including added features)
selected_features = [
    'HRxDuration',
    'Duration_Temp',
    'Duration',
    'Body_Temp',
    'Heart_Rate',
    'Age',
    'Body_Fat_Percentage_Est',
    'HR_per_Minute',
    'Age_BMI_Interaction',          # New feature
    'HeartRate_Temp_Interaction',   # New feature
    'Duration_Squared',             # New feature
    'Age_Squared',                  # New feature
    'Weight',                       # Added Weight
    'Height',                       # Added Height
    'BMI',                          # Added BMI
    'BMR',                          # Added BMR
    'Sex_encoded'                   # Added Sex_encoded
]

# Log transform the target variable
X = df[selected_features]
y = np.log1p(df['Calories'])


# Data splitting
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# Define and train XGBoost model (attempting hyperparameter tuning)
base_score = y_train.mean()

model = XGBRegressor(
    n_estimators=2000,          # Increased n_estimators
    learning_rate=0.03,         # Decreased learning_rate
    max_depth=6,                # Adjusted max_depth
    subsample=0.7,              # Adjusted subsample
    colsample_bytree=0.7,       # Adjusted colsample_bytree
    gamma=0.1,                  # Added gamma
    reg_alpha=0.005,            # Added L1 regularization
    reg_lambda=0.8,             # Added L2 regularization
    random_state=42,
    early_stopping_rounds=100,  # Increased early_stopping_rounds
    eval_metric='rmsle',
    base_score=base_score,
    n_jobs=-1                   # Use all cores
)

print("Starting model training")
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=100)
print("Model training completed")


# Evaluate validation performance
val_preds_log = model.predict(X_val)
val_preds = np.expm1(val_preds_log) # Transform back from log scale
val_y = np.expm1(y_val) # Transform back from log scale
rmsle = np.sqrt(mean_squared_log_error(val_y, val_preds))
print(f"Validation RMSLE: {rmsle:.4f}")


from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_log_error

# 추가 파생변수 생성 함수
def add_advanced_features(df):
    df = df.copy()
    # 로그, 제곱근, 역수, 다항식, 상호작용
    for col in ['BMI', 'Duration', 'Heart_Rate', 'Body_Temp', 'Age']:
        df[f'{col}_log'] = np.log1p(df[col])
        df[f'{col}_sqrt'] = np.sqrt(df[col])
        df[f'{col}_inv'] = 1 / (df[col] + 1e-3)
        df[f'{col}_squared'] = df[col] ** 2
    # 상호작용
    df['BMI_Age'] = df['BMI'] * df['Age']
    df['BMI_HeartRate'] = df['BMI'] * df['Heart_Rate']
    df['Duration_HeartRate'] = df['Duration'] * df['Heart_Rate']
    df['Temp_HeartRate'] = df['Body_Temp'] * df['Heart_Rate']
    return df

df = add_advanced_features(df)
test_df = add_advanced_features(test_df)

# feature 목록 확장
enhanced_features = selected_features + [
    f'{col}_{suffix}'
    for col in ['BMI', 'Duration', 'Heart_Rate', 'Body_Temp', 'Age']
    for suffix in ['log', 'sqrt', 'inv', 'squared']
] + [
    'BMI_Age', 'BMI_HeartRate', 'Duration_HeartRate', 'Temp_HeartRate'
]

X = df[enhanced_features]
y = np.log1p(df['Calories'])
X_test_pred = test_df[enhanced_features]

# KFold Cross-Validation 및 XGBoost 튜닝
kf = KFold(n_splits=5, shuffle=True, random_state=42)
val_scores = []
models = []
for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    model = XGBRegressor(
        n_estimators=3000,
        learning_rate=0.02,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.2,
        reg_alpha=0.01,
        reg_lambda=1.0,
        random_state=fold,
        early_stopping_rounds=200,
        eval_metric='rmsle',
        n_jobs=-1
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=0)
    val_pred = np.expm1(model.predict(X_val))
    val_true = np.expm1(y_val)
    rmsle = np.sqrt(mean_squared_log_error(val_true, val_pred))
    print(f"Fold {fold+1} RMSLE: {rmsle:.4f}")
    val_scores.append(rmsle)
    models.append(model)
print(f"\nAverage CV RMSLE: {np.mean(val_scores):.4f}")

# 앙상블 예측
preds = np.zeros(X_test_pred.shape[0])
for model in models:
    preds += np.expm1(model.predict(X_test_pred)) / len(models)
test_df['Predicted_Calories'] = preds
print(test_df[['Predicted_Calories']].head())


# Evaluate validation performance
val_preds_log = model.predict(X_val)
val_preds = np.expm1(val_preds_log) # Transform back from log scale
val_y = np.expm1(y_val) # Transform back from log scale
rmsle = np.sqrt(mean_squared_log_error(val_y, val_preds))
print(f"Validation RMSLE: {rmsle:.4f}")


# Evaluate training data performance (train_score)
train_preds_log = model.predict(X_train)
train_preds = np.expm1(train_preds_log)
train_y = np.expm1(y_train)
train_rmsle = np.sqrt(mean_squared_log_error(train_y, train_preds))
print(f"Training RMSLE: {train_rmsle:.4f}")


import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
# Training loss (train_score) is model.evals_result()['validation_0']['rmsle']
# Validation loss (test_score) is model.evals_result()['validation_1']['rmsle']
plt.plot(model.evals_result()['validation_0']['rmsle'], label='Train RMSLE') # Check model.evals_result()
plt.title('Model Learning Curve (RMSLE)')
plt.xlabel('Epochs')
plt.ylabel('RMSLE')
plt.legend()
plt.grid(True)
plt.show()


df_pred = test_df.copy()
X_test_pred = df_pred[enhanced_features]

test_preds_log = model.predict(X_test_pred)

pred = np.expm1(test_preds_log)

df_pred['Predicted_Calories'] = pred

print("df_pred with 'Predicted_Calories' column:")
print(df_pred.head())


id = range(750000, 1000000)

df_submit = pd.DataFrame({
    'id': id,
    'Calories': df_pred['Predicted_Calories']
})

print(df_submit.head())

df_submit.to_csv('submission.csv', index=False)

print("Submission file 'submission.csv' created successfully.")

