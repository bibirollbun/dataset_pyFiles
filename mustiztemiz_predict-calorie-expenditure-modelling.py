


# This Python 3 environment comaes with many helpful analytics libraries installed
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


# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
import xgboost as XGBRegressor
import lightgbm as lgb
import catboost as cb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectKBest, f_regression


# Load datasets
train_df = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')
sample_submission_df = pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv')


train_df


# Pairplot for numerical features
numerical_features = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp', 'Calories']
sns.pairplot(train_df[numerical_features], diag_kind='kde', corner=True)
plt.suptitle('Pairplot of Numerical Features', y=1.02)
plt.show()


# Check for outliers using boxplots
plt.figure(figsize=(12, 8))
train_df[numerical_features].boxplot()
plt.title('Boxplot of Numerical Features')
plt.xticks(rotation=45)
plt.show()


# Load datasets
original_train_df = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')
sample_submission_df = pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv')

# Apply outlier removal to create cleaned training data
numerical_features = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp', 'Calories']
train_df = original_train_df.copy()

# Remove outliers using IQR method
for feature in numerical_features:
    Q1 = train_df[feature].quantile(0.25)
    Q3 = train_df[feature].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    train_df = train_df[~((train_df[feature] < lower_bound) | (train_df[feature] > upper_bound))]

print(f"Original training data shape: {original_train_df.shape}")
print(f"Training data shape after outlier removal: {train_df.shape}")


# Preprocessing
# Encode categorical variables
train_df['Sex'] = LabelEncoder().fit_transform(train_df['Sex'])
test_df['Sex'] = LabelEncoder().fit_transform(test_df['Sex'])


# Preprocessing
# Encode categorical variables
train_df['Sex'] = LabelEncoder().fit_transform(train_df['Sex'])
test_df['Sex'] = LabelEncoder().fit_transform(test_df['Sex'])

# Select features and target
features = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp', 'Sex']
X = train_df[features]
y = train_df['Calories']
X_test = test_df[features]

# Feature Engineering (optional)
# Create interaction features
X['Weight_Height_Ratio'] = X['Weight'] / (X['Height'] + 1e-3)  # Avoid division by zero
X['BMI'] = X['Weight'] / ((X['Height']/100) ** 2)
X['Heart_Rate_Age_Ratio'] = X['Heart_Rate'] / (X['Age'] + 1e-3)
X['Intensity'] = X['Heart_Rate'] * X['Duration']

# Apply same transformations to test data
X_test['Weight_Height_Ratio'] = X_test['Weight'] / (X_test['Height'] + 1e-3)
X_test['BMI'] = X_test['Weight'] / ((X_test['Height']/100) ** 2)
X_test['Heart_Rate_Age_Ratio'] = X_test['Heart_Rate'] / (X_test['Age'] + 1e-3)
X_test['Intensity'] = X_test['Heart_Rate'] * X_test['Duration']

# Feature Selection using SelectKBest (optional)
selector = SelectKBest(f_regression, k='all')
selector.fit(X, y)
# Print feature scores
feature_scores = pd.DataFrame({'Feature': X.columns, 'Score': selector.scores_})
print("Feature Importance:")
print(feature_scores.sort_values(by='Score', ascending=False))

# Scale numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# Split the training data for validation
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)



# -------------------------
# 1. Veriyi HazÄ±rla
# -------------------------
X = cleaned_train_df.drop(columns=['Calories', 'id'])
y = cleaned_train_df['Calories']
X_test_clean = cleaned_test_df.drop(columns=['id'])
test_ids = cleaned_test_df['id']

# -------------------------
# 2. EÄŸitim ve DoÄŸrulama Seti
# -------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)



import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Modeller
import xgboost as xgb
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import lightgbm as lgb



# 2. Modelleri TanÄ±mla
# -------------------------
models = {
    'XGBoost': xgb.XGBRegressor(
        objective='reg:squarederror',
        colsample_bytree=0.3,
        learning_rate=0.1,
        max_depth=5,
        alpha=10,
        n_estimators=1000,
        random_state=42
    ),
    'LightGBM': LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.1,
        max_depth=5,
        colsample_bytree=0.7,
        subsample=0.9,
        random_state=42
    ),
    'CatBoost': CatBoostRegressor(
        iterations=1000,
        learning_rate=0.1,
        depth=5,
        verbose=100,
        random_seed=42
    )
}

# -------------------------
# 3. EÄŸit ve Tahmin Et
# -------------------------
submissions = {}

for name, model in models.items():
    print(f"\nğŸ§  Training {name}...\n")
    
    if name == 'CatBoost':
        model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50)
    
    elif name == 'LightGBM':
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='mae',
            callbacks=[
                lgb.early_stopping(50),
                lgb.log_evaluation(100)
            ]
        )
    
    else:  # XGBoost
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='mae',
            early_stopping_rounds=50,
            verbose=200
        )
    
    preds = model.predict(X_test)
    val_preds = model.predict(X_val)
    
    mae = mean_absolute_error(y_val, val_preds)
    print(f"âœ… {name} Validation MAE: {mae:.4f}")
    
    submissions[name] = np.clip(preds, 0, None)

# -------------------------
# 4. Submission DosyalarÄ±
# -------------------------
for name, preds in submissions.items():
    sample_submission_df = pd.DataFrame({
        'id': test_df['id'],
        'Calories': preds
    })
    filename = f'submission_{name}.csv'
    sample_submission_df.to_csv(filename, index=False)
    print(f"ğŸ“� Saved: {filename}")




