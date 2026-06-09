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


#data handling
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import math


train_set = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
test_set = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')
sample_set = pd.read_csv('/kaggle/input/playground-series-s5e10/sample_submission.csv')


print(f"Shape of train set: {train_set.shape}")
print(f"Shape of test set: {test_set.shape}")


train_df = train_set.copy()
test_df = test_set.copy()


print(f"Train Set : \n{train_df.head()}")
print(f"Test Set : \n{test_df.head()}")


y_train = train_df['accident_risk']


train_df.head()


train_df.info()


train_df.columns


#Drop id column
train_df = train_df.drop('id', axis=1)
test_df = test_df.drop('id', axis=1)


#numeric features
numeric_cols = train_df.select_dtypes(include=['number']).columns.tolist()
print(numeric_cols)


#Converting Boolean dtype into 0's and 1's form
train_df[train_df.select_dtypes(bool).columns] = train_df.select_dtypes(bool).astype(int)
test_df[test_df.select_dtypes(bool).columns] = test_df.select_dtypes(bool).astype(int)


#categorical feature
categorical_cols = train_df.select_dtypes(include=['object']).columns.tolist()
print(categorical_cols)


train_df[categorical_cols].nunique()


train_df.head()


#Check for Duplicated value
train_df.duplicated().sum()


train_df = train_df.drop_duplicates()


train_df.duplicated().sum()


train_df.isna().sum()


test_df.isna().sum()


train_df.describe()


print(f"Shape of train_df: {train_df.shape}")
print(f"Shape of test_df: {test_df.shape}")


# Target analysis
target_col = 'accident_risk'
if target_col not in train_df.columns:
    print("Target column not exist in dataset")
else:
    plt.figure(figsize=(8,5))
    
    sns.histplot(train_df[target_col], bins=20, kde=True, color="skyblue", edgecolor="black")
    
    # Formatting
    plt.title("Distribution of Survival Probability (0â€“1)")
    plt.xlabel("Survival Probability")
    plt.ylabel("Count")
    plt.xlim(0, 1)   # since range is 0â€“1
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    plt.show()


#Categorical feature analysis



# Select categorical columns
#categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns

# Define grid size (e.g., 2 rows Ã— 3 columns)
n_cols = 3
n_rows = -(-len(categorical_cols) // n_cols)  # Ceiling division

# Create subplots
fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))

# Flatten axes for easy iteration
axes = axes.flatten()

# Plot each categorical feature as a pie chart
for i, col in enumerate(categorical_cols):
    counts = train_df[col].value_counts(dropna=False)
    axes[i].pie(
        counts,
        labels=counts.index.astype(str),
        autopct='%1.1f%%',
        startangle=90,
        wedgeprops={'edgecolor': 'white'}
    )
    axes[i].set_title(f'{col} Distribution')

# Hide any unused subplots
for j in range(i+1, len(axes)):
    axes[j].axis('off')

plt.tight_layout()
plt.show()



#Analysing numeric columns

#numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
n_cols = 2
n_rows = math.ceil(len(numeric_cols))

plt.figure(figsize=(10, 4*n_rows))
gs = gridspec.GridSpec(n_rows, n_cols)

for i, col in enumerate(numeric_cols):
    ax0 = plt.subplot(gs[i, 0])
    sns.histplot(train_df[col], kde=True, bins=30, ax=ax0)
    ax0.set_title(f'Histogram of {col}')

    ax1 = plt.subplot(gs[i, 1])
    sns.boxplot(x=train_df[col], ax=ax1)
    ax1.set_title(f'Boxplot of {col}')

plt.tight_layout()
plt.show()






# feature engineering for numeric features
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# === 1ï¸�âƒ£ Log Transformations ===
for df in [train_df, test_df]:
    df['log_accidents'] = np.log1p(df['num_reported_accidents'])
    df['curvature_log'] = np.log1p(df['curvature'])

# === 2ï¸�âƒ£ Interaction Features ===
for df in [train_df, test_df]:
    df['curvature_x_speed'] = df['curvature'] * df['speed_limit']
    df['accidents_per_lane'] = df['num_reported_accidents'] / df['num_lanes']
    df['speed_per_lane'] = df['speed_limit'] / df['num_lanes']

# === 3ï¸�âƒ£ Binning ===
for df in [train_df, test_df]:
    df['speed_cat'] = pd.cut(df['speed_limit'], bins=[0, 40, 80, 120],
                             labels=['Low', 'Medium', 'High'])

# === 4ï¸�âƒ£ Scaling (Fit only on train, transform both) ===
scaler = StandardScaler()
scaled_cols = ['num_lanes', 'curvature', 'speed_limit', 'num_reported_accidents']

train_df[scaled_cols] = scaler.fit_transform(train_df[scaled_cols])
test_df[scaled_cols] = scaler.transform(test_df[scaled_cols])



# feature enginerring for categorical columns
import pandas as pd

# Define your categorical columns
#categorical_cols = ['road_type', 'lighting', 'weather', 'time_of_day']



# ğŸ§  2ï¸�âƒ£ Apply one-hot encoding
train_encoded = pd.get_dummies(train_df[categorical_cols], prefix=categorical_cols, drop_first=True)
test_encoded = pd.get_dummies(test_df[categorical_cols], prefix=categorical_cols, drop_first=True)

# ğŸ§  3ï¸�âƒ£ Align both dataframes (ensure same columns)
train_encoded, test_encoded = train_encoded.align(test_encoded, join='left', axis=1, fill_value=0)

# ğŸ§  4ï¸�âƒ£ Concatenate encoded features back to the main datasets
train_df = pd.concat([train_df, train_encoded], axis=1)
test_df = pd.concat([test_df, test_encoded], axis=1)

# ğŸ§  5ï¸�âƒ£ Optionally, drop original categorical columns
train_df = train_df.drop(columns=categorical_cols)
test_df = test_df.drop(columns=categorical_cols)

print("âœ… One-hot encoding done successfully!")
print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)



final_train_df = train_df.copy()
final_test_df = test_df.copy()


final_train_df[final_train_df.select_dtypes(bool).columns] = final_train_df.select_dtypes(bool).astype(int)
final_test_df[final_test_df.select_dtypes(bool).columns] = final_test_df.select_dtypes(bool).astype(int)


# Map ordered categories to numeric
mapping = {'Low': 0, 'Medium': 1, 'High': 2}

final_train_df['speed_cat_num'] = train_df['speed_cat'].map(mapping)
final_test_df['speed_cat_num'] = test_df['speed_cat'].map(mapping)



final_train_df = final_train_df.drop(columns=['speed_cat'])
final_test_df = final_test_df.drop(columns=['speed_cat'])



#Creating y tain as a target 
final_y_train = final_train_df['accident_risk']
final_train_df = final_train_df.drop(columns=['accident_risk'])


final_train_df.head()


final_train_df.shape


final_test_df.shape


final_train_df.columns.equals(final_test_df.columns)


from sklearn.model_selection import train_test_split

# Features and target
X = final_train_df.copy()
y = final_y_train.copy()

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,       # 20% for testing
    random_state=42,     # reproducibility
    shuffle=True         # shuffle rows before splitting
)

# Check shapes
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)




for col in X.select_dtypes(['category']).columns:
    X[col] = X[col].cat.codes    


# Cross-Validation with XGBoost
import xgboost as xgb
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
#from sklearn.datasets import load_boston
import numpy as np

# 1. Load data
#data = load_boston()
#X, y = data.data, data.target

# 2. Define base model
model = xgb.XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    objective='reg:squarederror',
    enable_categorical=True
)

# 3. K-Fold Cross Validation (scikit-learn)
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, scoring="neg_root_mean_squared_error", cv=kfold)
print("Cross-Validation RMSE per fold:", -scores)
print("Mean RMSE:", -scores.mean())

# 4. XGBoostâ€™s built-in CV
dtrain = xgb.DMatrix(X, label=y)
params = {
    "objective": "reg:squarederror",
    "learning_rate": 0.05,
    "max_depth": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eval_metric": "rmse",
    "enable_categorical": True
}
cv_results = xgb.cv(
    dtrain=dtrain,
    params=params,
    nfold=5,
    num_boost_round=500,
    early_stopping_rounds=30,
    metrics="rmse",
    seed=42
)
print("\nXGBoost Built-in CV Results:")
print(cv_results.tail())
print("Best RMSE:", cv_results['test-rmse-mean'].min())

# 5. Hyperparameter tuning with GridSearchCV
param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [200, 500, 800]
}
grid = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring='neg_root_mean_squared_error',
    cv=5,
    verbose=1,
    n_jobs=-1
)
grid.fit(X, y)
print("\nBest Parameters:", grid.best_params_)
print("Best RMSE from Grid Search:", -grid.best_score_)



grid.best_params_


#Training final model
final_model = xgb.XGBRegressor(learning_rate= 0.05,
                                max_depth=7,
                                n_estimators=200,
                               random_state=42,
                                objective='reg:squarederror',
                                enable_categorical=True
                              )

final_model.fit(X,y)


prediction = final_model.predict(final_test_df)
Submission = pd.DataFrame({
    'ID': test_set['id'],
    'accident_risk': prediction
})
Submission.to_csv('submission.csv', index=False)
Submission.head(10)

